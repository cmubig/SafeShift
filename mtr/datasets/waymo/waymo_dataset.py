# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import os
import numpy as np
from pathlib import Path
import pickle
import torch
from tqdm import tqdm

from mtr.datasets.dataset import DatasetTemplate
from mtr.utils import common_utils
from mtr.config import cfg
import time


class WaymoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training=True, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, training=training, logger=logger)
        self.data_root = cfg.ROOT_DIR / self.dataset_cfg.DATA_ROOT
        self.data_path = self.data_root / self.dataset_cfg.SPLIT_DIR[self.mode]

        self.infos = self.get_all_infos(self.data_root / self.dataset_cfg.INFO_FILE[self.mode])
        self.logger.info(f'Total scenes after filters: {len(self.infos)}')

        self.use_frenet = self.dataset_cfg.get('USE_FRENET', False)
        self.frenet_path = self.data_root / self.dataset_cfg.get('FRENET_BASE', 'joint_frenet')
        self.append_both = self.dataset_cfg.get('APPEND_BOTH', False)
        assert (not self.append_both) or (not self.use_frenet), 'Cannot use frenet primarily when appending both'

        self.load_mcnn = self.dataset_cfg.get('LOAD_MCNN', False)
        self.mcnn_root = self.dataset_cfg.get('MCNN_ROOT', '')
        self.mcnn_path = cfg.ROOT_DIR / self.mcnn_root
        
        self.weight_loss = self.dataset_cfg.get('WEIGHT_LOSS', False)
        self.fe_weight_input = self.dataset_cfg.get('FE_WEIGHT_INPUT', False)

        # if self.load_mcnn and (self.use_frenet or self.append_both):
        #     raise NotImplementedError('Still need to implement MotionCNN + Frenet stuff')

        test = self[0]
        # if self.load_mcnn:
        #     sus = np.random.RandomState(42)
        #     for i in tqdm(range(100)):
        #         item = self[int(len(self)*sus.rand())]
        #     breakpoint()
    
    def get_all_infos(self, info_path):
        self.logger.info(f'Start to load infos from {info_path}')
        with open(info_path, 'rb') as f:
            src_infos = pickle.load(f)

        infos = src_infos[::self.dataset_cfg.SAMPLE_INTERVAL[self.mode]]
        if 'BOOTSTRAP' in self.dataset_cfg:
            bootstrap_random = np.random.RandomState(self.dataset_cfg.BOOTSTRAP.idx)
            #infos = infos[self.dataset_cfg.BOOTSTRAP.idx::self.dataset_cfg.BOOTSTRAP.subsample_interval]
            infos = bootstrap_random.choice(infos, size=len(infos), replace=True).tolist()

        self.logger.info(f'Total scenes before filters: {len(infos)}')

        for func_name, val in self.dataset_cfg.INFO_FILTER_DICT.items():
            infos = getattr(self, func_name)(infos, val)

        return infos

    def filter_info_by_object_type(self, infos, valid_object_types=None):
        ret_infos = []
        for cur_info in infos:
            num_interested_agents = cur_info['tracks_to_predict']['track_index'].__len__()
            if num_interested_agents == 0:
                continue

            valid_mask = []
            for idx, cur_track_index in enumerate(cur_info['tracks_to_predict']['track_index']):
                valid_mask.append(cur_info['tracks_to_predict']['object_type'][idx] in valid_object_types)

            valid_mask = np.array(valid_mask) > 0
            if valid_mask.sum() == 0:
                continue

            assert len(cur_info['tracks_to_predict'].keys()) == 3, f"{cur_info['tracks_to_predict'].keys()}"
            cur_info['tracks_to_predict']['track_index'] = list(np.array(cur_info['tracks_to_predict']['track_index'])[valid_mask])
            cur_info['tracks_to_predict']['object_type'] = list(np.array(cur_info['tracks_to_predict']['object_type'])[valid_mask])
            cur_info['tracks_to_predict']['difficulty'] = list(np.array(cur_info['tracks_to_predict']['difficulty'])[valid_mask])

            ret_infos.append(cur_info)
        self.logger.info(f'Total scenes after filter_info_by_object_type: {len(ret_infos)}')
        return ret_infos

    def __len__(self):
        return len(self.infos)
    
    def total_tracks(self):
        return sum([len(x['tracks_to_predict']['track_index']) for x in self.infos])

    def __getitem__(self, index):
        # When appending both, need to concat:
        # - obj_trajs, last dim 29->31
        # - map_polylines, last_dim 9->11
        
        n_to_predict = len(self.infos[index]['tracks_to_predict']['track_index'])
        # TODO: update here, once scores are traj based rather than scene based
        if self.weight_loss:
            # Divide by 10 for stability
            #scores = np.array([self.infos[index]['score']]*n_to_predict).astype(np.float32) / 10
            scores = self.infos[index]['traj_scores_asym_combined'].astype(np.float32) / 10
        else:
            scores = np.array([1]*n_to_predict).astype(np.float32)
        if self.fe_weight_input:
            # Divide by 10 for stability
            #fe_scores = np.array([self.infos[index]['fe_score']]*n_to_predict).astype(np.float32) / 10
            fe_scores = self.infos[index]['traj_scores_fe'].astype(np.float32) / 10
        else:
            fe_scores = np.array([1]*n_to_predict).astype(np.float32)

        if self.load_mcnn:
            ret_infos = self.create_data_mcnn(index)
            ret_infos['scores'] = scores
            ret_infos['fe_scores'] = fe_scores
            original_infos = self.create_scene_level_data(index)
            ret_infos['obj_trajs_future_state'] = original_infos['obj_trajs_future_state']
            ret_infos['obj_trajs_future_mask'] = original_infos['obj_trajs_future_mask']
            ret_infos['track_index_to_predict'] = original_infos['track_index_to_predict']

            # Need to at least add in the sd_infos and such for self
            if self.use_frenet or self.append_both:
                old_frenet = self.use_frenet
                self.use_frenet = True
                sd_infos = self.create_scene_level_data(index) 
                sd_trajs_fut = np.stack([x[track_idx][..., :2] for x, track_idx in \
                                               zip(sd_infos['obj_trajs_future_state'], sd_infos['track_index_to_predict'])], axis=0)
                ret_infos['sd_trajs_fut'] = sd_trajs_fut
                self.use_frenet = old_frenet
            return ret_infos
        
        if not self.append_both:
            ret_infos = self.create_scene_level_data(index)
            ret_infos['scores'] = scores
            ret_infos['fe_scores'] = fe_scores
        else:
            old_frenet = self.use_frenet
            self.use_frenet = False
            ret_infos = self.create_scene_level_data(index)
            self.use_frenet = True
            sd_infos = self.create_scene_level_data(index)
            self.use_frenet = old_frenet
            ret_infos['obj_trajs'] = np.concatenate([ret_infos['obj_trajs'], sd_infos['obj_trajs'][..., :2]], axis=-1)
            ret_infos['obj_trajs_future_state'] = np.concatenate([ret_infos['obj_trajs_future_state'], 
                                                                  sd_infos['obj_trajs_future_state'][..., :2]], axis=-1)
            ret_infos['map_polylines'] = np.concatenate([ret_infos['map_polylines'], sd_infos['map_polylines'][..., :2]], axis=-1)
            ret_infos['scores'] = scores
            ret_infos['fe_scores'] = fe_scores
            
        return ret_infos
    
    def create_data_mcnn(self, index):
        info = self.infos[index]
        scene_id = info['scenario_id']
        start_time = time.time()
        with open(self.data_path / f'sample_{scene_id}.pkl', 'rb') as f:
            info = pickle.load(f)
        obj_ids_to_predict = np.array(info['track_infos']['object_id'])[info['tracks_to_predict']['track_index']]
        #print('Load time scenario: ', time.time() - start_time)

        start_time = time.time()
        scene_path = os.path.join(self.mcnn_path, scene_id)
        agent_paths = [os.path.join(scene_path, 'agent_data', f'{obj_id}.npz') for obj_id in obj_ids_to_predict]
        assert all([os.path.exists(x) for x in agent_paths]), 'Missing to_predict agent file'

        # No need to load roadgraph
        #roadgraph_data = np.load(os.path.join(scene_path, 'roadgraph_data', 'segments_global.npz'))['roadgraph_segments']

        datas = [dict(np.load(agent_path, allow_pickle=True)) for agent_path in agent_paths]
        #print('Load time agent: ', time.time() - start_time)

        for data_idx, data in enumerate(datas):
            data['raster'] = data['raster'].transpose(2, 0, 1)/255
            data['scenario_id'] = data['scenario_id'].item()
            # Needs to have xy in first 2, then yaw in index 6
            obj_trajs_full = info['track_infos']['trajs']  # (num_objects, num_timestamp, 10)
            data['center_objects_world'] = np.array([*data['current_xy_global'], 0, 0, 0, 0, data['yaw'], 0, 0, 0])
            data['center_gt_trajs_src'] = obj_trajs_full[info['tracks_to_predict']['track_index'][data_idx]]

        start_time = time.time()
        data_out = {}
        for k in datas[0].keys():
            #data_out[k] = manual_stack([data[k] for data in datas])
            data_out[k] = np.stack([data[k] for data in datas])

        # Needed for collate_batch & generate_prediction_dicts
        track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
        data_out['track_index_to_predict'] = track_index_to_predict
        #'center_objects_id': np.array(track_infos['object_id'])[track_index_to_predict],
        #'center_objects_type': np.array(track_infos['object_type'])[track_index_to_predict],
        data_out['center_objects_id'] = np.array(info['track_infos']['object_id'])[track_index_to_predict]
        data_out['center_objects_type'] = np.array(info['track_infos']['object_type'])[track_index_to_predict]

        #print('Stack time: ', time.time() - start_time)
        return data_out
    
    def create_scene_level_data(self, index):
        """
        Args:
            index (index):

        Returns:

        """
        info = self.infos[index]
        scene_id = info['scenario_id']
        with open(self.data_path / f'sample_{scene_id}.pkl', 'rb') as f:
            info = pickle.load(f)
        
        if self.use_frenet:
            with open(self.frenet_path / f'frenet_{scene_id}.pkl', 'rb') as f:
                frenet_info = pickle.load(f)
        else:
            frenet_info = None

        sdc_track_index = info['sdc_track_index']
        current_time_index = info['current_time_index']
        timestamps = np.array(info['timestamps_seconds'][:current_time_index + 1], dtype=np.float32)

        track_infos = info['track_infos']

        track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
        obj_types = np.array(track_infos['object_type'])
        obj_ids = np.array(track_infos['object_id'])
        obj_trajs_full = track_infos['trajs']  # (num_objects, num_timestamp, 10)
        obj_trajs_past = obj_trajs_full[:, :current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1:]

        center_objects, track_index_to_predict = self.get_interested_agents(
            track_index_to_predict=track_index_to_predict,
            obj_trajs_full=obj_trajs_full,
            current_time_index=current_time_index,
            obj_types=obj_types, scene_id=scene_id
        )

        # Start here, convert center_objs to Frenet
        if self.use_frenet:
            frenet_trajs = frenet_info['trajs']
            frenet_center_objs = np.stack([frenet_trajs[i, to_pred] for i, to_pred in enumerate(track_index_to_predict)])
            frenet_center_objs = frenet_center_objs[:, current_time_index]
            # patch in xy -> sd
            center_objects[..., :2] = frenet_center_objs[..., :2]
            # patch in vx, vy -> sd_vel
            center_objects[..., 7:9] = frenet_center_objs[..., 2:4]
            # patch in heading -> lane deflection heading
            center_objects[..., 6] = frenet_center_objs[..., 4]
        
        # Now, everything here is in Frenet as well
        (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs,
            center_gt_trajs_mask, center_gt_final_valid_idx,
            track_index_to_predict_new, sdc_track_index_new, obj_types, obj_ids) = self.create_agent_data_for_center_objects(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past, obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict, sdc_track_index=sdc_track_index,
            timestamps=timestamps, obj_types=obj_types, obj_ids=obj_ids, frenet_info=frenet_info
        )

        # Everything here has been converted except for center_gt_trajs_src, which is only used in generate_prediction_dicts
        ret_dict = {
            'scenario_id': np.array([scene_id] * len(track_index_to_predict)),
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            'track_index_to_predict': track_index_to_predict_new,  # used to select center-features
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,
            'obj_types': obj_types,
            'obj_ids': obj_ids,

            'center_objects_world': center_objects,
            'center_objects_id': np.array(track_infos['object_id'])[track_index_to_predict],
            'center_objects_type': np.array(track_infos['object_type'])[track_index_to_predict],

            'obj_trajs_future_state': obj_trajs_future_state,
            'obj_trajs_future_mask': obj_trajs_future_mask,
            'center_gt_trajs': center_gt_trajs,
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            'center_gt_trajs_src': obj_trajs_full[track_index_to_predict]
        }

        # TODO: make sure frenet_lanes get passed into this; center_gt_trajs_src is enough for recovery assessment
        # Need to pad to the largest that's in the entire dataset lol
        if self.use_frenet:
            # Need to pad
            frenet_lanes = frenet_info['lanes']
            lane_lengths = np.array([lane.shape[0] for lane in frenet_lanes], dtype=np.int)
            # From looking ahead of time, longest lane is 2500 long
            max_lane = 2500
            padded_lanes = np.zeros((len(frenet_lanes), max_lane, 3), dtype=np.float32)
            for i, lane in enumerate(frenet_lanes):
                padded_lanes[i, :len(lane)] = lane
            
            ret_dict['frenet_lanes'] = padded_lanes
            ret_dict['frenet_lane_lengths'] = lane_lengths

        if not self.dataset_cfg.get('WITHOUT_HDMAP', False):
            if info['map_infos']['all_polylines'].__len__() == 0:
                info['map_infos']['all_polylines'] = np.zeros((2, 7), dtype=np.float32)
                print(f'Warning: empty HDMap {scene_id}')

            map_polylines_data, map_polylines_mask, map_polylines_center = self.create_map_data_for_center_objects(
                center_objects=center_objects, map_infos=info['map_infos'],
                center_offset=self.dataset_cfg.get('CENTER_OFFSET_OF_MAP', (30.0, 0)),
                frenet_info=frenet_info
            )   # (num_center_objects, num_topk_polylines, num_points_each_polyline, 9), (num_center_objects, num_topk_polylines, num_points_each_polyline)

            ret_dict['map_polylines'] = map_polylines_data
            ret_dict['map_polylines_mask'] = (map_polylines_mask > 0)
            ret_dict['map_polylines_center'] = map_polylines_center

        return ret_dict

    def create_agent_data_for_center_objects(
            self, center_objects, obj_trajs_past, obj_trajs_future, track_index_to_predict, sdc_track_index, timestamps,
            obj_types, obj_ids, frenet_info=None
        ):
        obj_trajs_data, obj_trajs_mask, obj_trajs_future_state, obj_trajs_future_mask = self.generate_centered_trajs_for_agents(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past,
            obj_types=obj_types, center_indices=track_index_to_predict,
            sdc_index=sdc_track_index, timestamps=timestamps, obj_trajs_future=obj_trajs_future,
            frenet_info=frenet_info
        )

        # generate the labels of track_objects for training
        center_obj_idxs = np.arange(len(track_index_to_predict))
        center_gt_trajs = obj_trajs_future_state[center_obj_idxs, track_index_to_predict]  # (num_center_objects, num_future_timestamps, 4)
        center_gt_trajs_mask = obj_trajs_future_mask[center_obj_idxs, track_index_to_predict]  # (num_center_objects, num_future_timestamps)
        center_gt_trajs[center_gt_trajs_mask == 0] = 0

        # filter invalid past trajs
        assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
        valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)  # (num_objects (original))

        obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps)
        obj_trajs_data = obj_trajs_data[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps, C)
        obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps_future, 4):  [x, y, vx, vy]
        obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]  # (num_center_objects, num_objects, num_timestamps_future):
        obj_types = obj_types[valid_past_mask]
        obj_ids = obj_ids[valid_past_mask]

        valid_index_cnt = valid_past_mask.cumsum(axis=0)
        track_index_to_predict_new = valid_index_cnt[track_index_to_predict] - 1
        sdc_track_index_new = valid_index_cnt[sdc_track_index] - 1  # TODO: CHECK THIS

        assert obj_trajs_future_state.shape[1] == obj_trajs_data.shape[1]
        assert len(obj_types) == obj_trajs_future_mask.shape[1]
        assert len(obj_ids) == obj_trajs_future_mask.shape[1]

        # generate the final valid position of each object
        obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, :, k] > 0  # (num_center_objects, num_objects)
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

        center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
        for k in range(center_gt_trajs_mask.shape[1]):
            cur_valid_mask = center_gt_trajs_mask[:, k] > 0  # (num_center_objects)
            center_gt_final_valid_idx[cur_valid_mask] = k

        return (obj_trajs_data, obj_trajs_mask > 0, obj_trajs_pos, obj_trajs_last_pos,
            obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask, center_gt_final_valid_idx,
            track_index_to_predict_new, sdc_track_index_new, obj_types, obj_ids)

    def get_interested_agents(self, track_index_to_predict, obj_trajs_full, current_time_index, obj_types, scene_id):
        center_objects_list = []
        track_index_to_predict_selected = []

        for k in range(len(track_index_to_predict)):
            obj_idx = track_index_to_predict[k]

            assert obj_trajs_full[obj_idx, current_time_index, -1] > 0, f'obj_idx={obj_idx}, scene_id={scene_id}'

            center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
            track_index_to_predict_selected.append(obj_idx)

        center_objects = np.stack(center_objects_list, axis=0)  # (num_center_objects, num_attrs)
        track_index_to_predict = np.array(track_index_to_predict_selected)
        return center_objects, track_index_to_predict

    @staticmethod
    def transform_trajs_to_center_coords(obj_trajs, center_xyz, center_heading, heading_index,
                                         rot_vel_index=None, do_rotation=True, frenet_info=None):
        """
        Args:
            obj_trajs (num_objects, num_timestamps, num_attrs):
                first three values of num_attrs are [x, y, z] or [x, y]
            center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
            center_heading (num_center_objects):
            heading_index: the index of heading angle in the num_attr-axis of obj_trajs
        """
        num_objects, num_timestamps, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]

        obj_trajs = obj_trajs.clone().view(1, num_objects, num_timestamps, num_attrs).repeat(num_center_objects, 1, 1, 1)

        if frenet_info is not None:
            frenet_trajs = frenet_info['trajs']
            is_hist = obj_trajs.shape[-2] == 11
            frenet_trajs = frenet_trajs[:, :, :11] if is_hist else frenet_trajs[:, :, 11:]
            frenet_trajs = torch.from_numpy(frenet_trajs)
            # patch in xy -> sd
            obj_trajs[..., :2] = frenet_trajs[..., :2]
            # patch in vx, vy -> sd_vel
            obj_trajs[..., 7:9] = frenet_trajs[..., 2:4]
            # patch in heading -> lane deflection heading
            obj_trajs[..., 6] = frenet_trajs[..., 4]

        # TODO: agent centric remove?
        obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
        if do_rotation:
        # if True:
            obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
                points=obj_trajs[:, :, :, 0:2].view(num_center_objects, -1, 2),
                angle=-center_heading
            ).view(num_center_objects, num_objects, num_timestamps, 2)

            obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

        # rotate direction of velocity
        if rot_vel_index is not None and do_rotation:
        # if rot_vel_index is not None and True:
            assert len(rot_vel_index) == 2
            obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
                points=obj_trajs[:, :, :, rot_vel_index].view(num_center_objects, -1, 2),
                angle=-center_heading
            ).view(num_center_objects, num_objects, num_timestamps, 2)

        return obj_trajs

    def generate_centered_trajs_for_agents(self, center_objects, obj_trajs_past, obj_types, center_indices, sdc_index, timestamps, obj_trajs_future, frenet_info=None):
        """[summary]

        Args:
            center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            obj_trajs_past (num_objects, num_timestamps, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            obj_types (num_objects):
            center_indices (num_center_objects): the index of center objects in obj_trajs_past
            centered_valid_time_indices (num_center_objects), the last valid time index of center objects
            timestamps ([type]): [description]
            obj_trajs_future (num_objects, num_future_timestamps, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        Returns:
            ret_obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
            ret_obj_valid_mask (num_center_objects, num_objects, num_timestamps):
            ret_obj_trajs_future (num_center_objects, num_objects, num_timestamps_future, 4):  [x, y, vx, vy]
            ret_obj_valid_mask_future (num_center_objects, num_objects, num_timestamps_future):
        """
        assert obj_trajs_past.shape[-1] == 10
        assert center_objects.shape[-1] == 10
        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape
        # transform to cpu torch tensor
        center_objects = torch.from_numpy(center_objects).float()
        obj_trajs_past = torch.from_numpy(obj_trajs_past).float()
        timestamps = torch.from_numpy(timestamps)
        do_rotation = not self.use_frenet

        # transform coordinates to the centered objects
        obj_trajs = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8],
            do_rotation=do_rotation,
            frenet_info=frenet_info
        )

        ## generate the attributes for each object
        object_onehot_mask = torch.zeros((num_center_objects, num_objects, num_timestamps, 5))
        object_onehot_mask[:, obj_types == 'TYPE_VEHICLE', :, 0] = 1
        object_onehot_mask[:, obj_types == 'TYPE_PEDESTRAIN', :, 1] = 1  # TODO: CHECK THIS TYPO
        object_onehot_mask[:, obj_types == 'TYPE_CYCLIST', :, 2] = 1
        object_onehot_mask[torch.arange(num_center_objects), center_indices, :, 3] = 1
        object_onehot_mask[:, sdc_index, :, 4] = 1

        object_time_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
        object_time_embedding[:, :, torch.arange(num_timestamps), torch.arange(num_timestamps)] = 1
        object_time_embedding[:, :, torch.arange(num_timestamps), -1] = timestamps

        object_heading_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
        object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

        vel = obj_trajs[:, :, :, 7:9]  # (num_centered_objects, num_objects, num_timestamps, 2)
        vel_pre = torch.roll(vel, shifts=1, dims=2)
        acce = (vel - vel_pre) / 0.1  # (num_centered_objects, num_objects, num_timestamps, 2)
        acce[:, :, 0, :] = acce[:, :, 1, :]

        # TODO: consider adding in SD coordinates instead of replacing?
        ret_obj_trajs = torch.cat((
            obj_trajs[:, :, :, 0:6], 
            object_onehot_mask,
            object_time_embedding, 
            object_heading_embedding,
            obj_trajs[:, :, :, 7:9], 
            acce,
        ), dim=-1)

        ret_obj_valid_mask = obj_trajs[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps)  # TODO: CHECK THIS, 20220322
        ret_obj_trajs[ret_obj_valid_mask == 0] = 0

        ##  generate label for future trajectories
        obj_trajs_future = torch.from_numpy(obj_trajs_future).float()
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8],
            do_rotation=do_rotation,
            frenet_info=frenet_info
        )
        ret_obj_trajs_future = obj_trajs_future[:, :, :, [0, 1, 7, 8]]  # (x, y, vx, vy)
        ret_obj_valid_mask_future = obj_trajs_future[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps_future)  # TODO: CHECK THIS, 20220322
        ret_obj_trajs_future[ret_obj_valid_mask_future == 0] = 0

        return ret_obj_trajs.numpy(), ret_obj_valid_mask.numpy(), ret_obj_trajs_future.numpy(), ret_obj_valid_mask_future.numpy()

    @staticmethod
    def generate_batch_polylines_from_map(polylines, point_sampled_interval=1,
                                          vector_break_dist_thresh=1.0, num_points_each_polyline=20,
                                          original_break_idxs=None):
        """
        Args:
            polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]

        Returns:
            ret_polylines: (num_polylines, num_points_each_polyline, 7)
            ret_polylines_mask: (num_polylines, num_points_each_polyline)
        """
        point_dim = polylines.shape[-1]

        sampled_points = polylines[::point_sampled_interval]
        sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
        buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]), axis=-1) # [ed_x, ed_y, st_x, st_y]
        buffer_points[0, 2:4] = buffer_points[0, 0:2]

        # TODO: ensure break_idxs here are consistent between center objects
        if original_break_idxs is None:
            break_idxs = (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4], axis=-1) > vector_break_dist_thresh).nonzero()[0]
        else:
            break_idxs = original_break_idxs
        polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
        ret_polylines = []
        ret_polylines_mask = []

        def append_single_polyline(new_polyline):
            cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
            cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
            cur_polyline[:len(new_polyline)] = new_polyline
            cur_valid_mask[:len(new_polyline)] = 1
            ret_polylines.append(cur_polyline)
            ret_polylines_mask.append(cur_valid_mask)

        for k in range(len(polyline_list)):
            if polyline_list[k].__len__() <= 0:
                continue
            for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
                append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

        ret_polylines = np.stack(ret_polylines, axis=0)
        ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

        ret_polylines = torch.from_numpy(ret_polylines)
        ret_polylines_mask = torch.from_numpy(ret_polylines_mask)

        # # CHECK the results
        # polyline_center = ret_polylines[:, :, 0:2].sum(dim=1) / ret_polyline_valid_mask.sum(dim=1).float()[:, None]  # (num_polylines, 2)
        # center_dist = (polyline_center - ret_polylines[:, 0, 0:2]).norm(dim=-1)
        # assert center_dist.max() < 10
        return ret_polylines, ret_polylines_mask, break_idxs

    def create_map_data_for_center_objects(self, center_objects, map_infos, center_offset, frenet_info=None):
        """
        Args:
            center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            map_infos (dict):
                all_polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
            center_offset (2):, [offset_x, offset_y]
        Returns:
            map_polylines (num_center_objects, num_topk_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
            map_polylines_mask (num_center_objects, num_topk_polylines, num_points_each_polyline)
        """
        num_center_objects = center_objects.shape[0]
        do_rotation = not self.use_frenet

        # transform object coordinates by center objects
        def transform_to_center_coordinates(neighboring_polylines, neighboring_polyline_valid_mask, inner_shape=20, do_rotation=True):
            # TODO: agent centric remove?
            neighboring_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
            if do_rotation:
            # if True:
                neighboring_polylines[:, :, :, 0:2] = common_utils.rotate_points_along_z(
                    points=neighboring_polylines[:, :, :, 0:2].view(num_center_objects, -1, 2),
                    angle=-center_objects[:, 6]
                ).view(num_center_objects, -1, inner_shape, 2)
                neighboring_polylines[:, :, :, 3:5] = common_utils.rotate_points_along_z(
                    points=neighboring_polylines[:, :, :, 3:5].view(num_center_objects, -1, 2),
                    angle=-center_objects[:, 6]
                ).view(num_center_objects, -1, inner_shape, 2)

            # use pre points to map
            # (num_center_objects, num_polylines, num_points_each_polyline, num_feat)
            xy_pos_pre = neighboring_polylines[:, :, :, 0:2]
            xy_pos_pre = torch.roll(xy_pos_pre, shifts=1, dims=-2)
            xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
            neighboring_polylines = torch.cat((neighboring_polylines, xy_pos_pre), dim=-1)

            neighboring_polylines[neighboring_polyline_valid_mask == 0] = 0
            return neighboring_polylines, neighboring_polyline_valid_mask

        polylines = torch.from_numpy(map_infos['all_polylines'].copy())

        # Already in Frenet
        center_objects = torch.from_numpy(center_objects)

        if not self.use_frenet:
            batch_polylines, batch_polylines_mask, _ = self.generate_batch_polylines_from_map(
                polylines=polylines.numpy(), point_sampled_interval=self.dataset_cfg.get('POINT_SAMPLED_INTERVAL', 1),
                vector_break_dist_thresh=self.dataset_cfg.get('VECTOR_BREAK_DIST_THRESH', 1.0),
                num_points_each_polyline=self.dataset_cfg.get('NUM_POINTS_EACH_POLYLINE', 20),
            )  # (num_polylines, num_points_each_polyline, 7), (num_polylines, num_points_each_polyline)
        else:
            _, _, break_idxs = self.generate_batch_polylines_from_map(
                polylines=polylines.numpy(), point_sampled_interval=self.dataset_cfg.get('POINT_SAMPLED_INTERVAL', 1),
                vector_break_dist_thresh=self.dataset_cfg.get('VECTOR_BREAK_DIST_THRESH', 1.0),
                num_points_each_polyline=self.dataset_cfg.get('NUM_POINTS_EACH_POLYLINE', 20),
            )  # (num_polylines, num_points_each_polyline, 7), (num_polylines, num_points_each_polyline) 

            frenet_polylines = torch.from_numpy(frenet_info['all_polylines'])
            polylines = polylines.unsqueeze(0).repeat(num_center_objects, 1, 1)
            # TODO: patch in Frenet here
            # xy -> sd
            polylines[:, :, :2] = frenet_polylines[:, :, :2]
            # xy_dir -> sd_dir
            polylines[:, :, 3:5] = frenet_polylines[:, :, 2:]

            batch_out = []
            batch_mask_out = []
            for inner_polylines in polylines:
                batch_polylines, batch_polylines_mask, _ = self.generate_batch_polylines_from_map(
                    polylines=inner_polylines.numpy(), point_sampled_interval=self.dataset_cfg.get('POINT_SAMPLED_INTERVAL', 1),
                    vector_break_dist_thresh=self.dataset_cfg.get('VECTOR_BREAK_DIST_THRESH', 1.0),
                    num_points_each_polyline=self.dataset_cfg.get('NUM_POINTS_EACH_POLYLINE', 20),
                    original_break_idxs=break_idxs
                )  # (num_polylines, num_points_each_polyline, 7), (num_polylines, num_points_each_polyline)
                batch_out.append(batch_polylines)
                batch_mask_out.append(batch_polylines_mask)
            batch_polylines = torch.stack(batch_out)
            batch_polylines_mask = torch.stack(batch_mask_out)

        # collect a number of closest polylines for each center objects
        num_of_src_polylines = self.dataset_cfg.NUM_OF_SRC_POLYLINES

        n_polylines = len(batch_polylines) if not self.use_frenet else batch_polylines.shape[1]
        if n_polylines > num_of_src_polylines:
            if not self.use_frenet:
                polyline_center = batch_polylines[:, :, 0:2].sum(dim=1) / torch.clamp_min(batch_polylines_mask.sum(dim=1).float()[:, None], min=1.0)
            else:
                polyline_center = batch_polylines[:, :, :, 0:2].sum(dim=2) / torch.clamp_min(batch_polylines_mask.sum(dim=2).float()[:, :, None], min=1.0)
            center_offset_rot = torch.from_numpy(np.array(center_offset, dtype=np.float32))[None, :].repeat(num_center_objects, 1)
            if do_rotation:
            # if True:
                center_offset_rot = common_utils.rotate_points_along_z(
                    points=center_offset_rot.view(num_center_objects, 1, 2),
                    angle=center_objects[:, 6]
                ).view(num_center_objects, 2)

            pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot

            if not self.use_frenet:
                dist = (pos_of_map_centers[:, None, :] - polyline_center[None, :, :]).norm(dim=-1)  # (num_center_objects, num_polylines)
            else:
                dist = (pos_of_map_centers[:, None, :] - polyline_center[:, :, :]).norm(dim=-1)  # (num_center_objects, num_polylines)
            topk_dist, topk_idxs = dist.topk(k=num_of_src_polylines, dim=-1, largest=False)
            if not self.use_frenet:
                map_polylines = batch_polylines[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
                map_polylines_mask = batch_polylines_mask[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)
            else:
                map_polylines = torch.stack([batch_polylines[i, idxs] for i, idxs in enumerate(topk_idxs)])
                map_polylines_mask = torch.stack([batch_polylines_mask[i, idxs] for i, idxs in enumerate(topk_idxs)])
        else:
            if not self.use_frenet:
                map_polylines = batch_polylines[None, :, :, :].repeat(num_center_objects, 1, 1, 1)
                map_polylines_mask = batch_polylines_mask[None, :, :].repeat(num_center_objects, 1, 1)
            else:
                map_polylines = batch_polylines
                map_polylines_mask = batch_polylines_mask

        map_polylines, map_polylines_mask = transform_to_center_coordinates(
            neighboring_polylines=map_polylines,
            neighboring_polyline_valid_mask=map_polylines_mask,
            inner_shape=batch_polylines.shape[1] if not self.use_frenet else batch_polylines.shape[2],
            do_rotation=do_rotation
        )

        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].float()).sum(dim=-2)  # (num_center_objects, num_polylines, 3)
        map_polylines_center = temp_sum / torch.clamp_min(map_polylines_mask.sum(dim=-1).float()[:, :, None], min=1.0)  # (num_center_objects, num_polylines, 3)

        map_polylines = map_polylines.numpy()
        map_polylines_mask = map_polylines_mask.numpy()
        map_polylines_center = map_polylines_center.numpy()

        return map_polylines, map_polylines_mask, map_polylines_center

    # Copied from tools.scenario_identification.frenet_interp ; path is weird
    def simple_frenet2cartesian(self, points, lane):
        points = points[:, :2].astype(np.float64)
        lane = lane[:, :2].astype(np.float64)

        ref_lane = lane
        ref_lane_segments = np.array([np.array(x) for x in zip(ref_lane[:-1], ref_lane[1:])])
        segment_lengths = np.linalg.norm(ref_lane_segments[:, 1] - ref_lane_segments[:, 0], axis=-1)
        # Frenet s for being at the start of each segment
        segment_frenet_s = np.array([0, *np.cumsum(segment_lengths)[:-1]])
        segment_idxs = np.array([0 if point[0] <= 0 else \
                                len(segment_frenet_s) - 1 if point[0] >= segment_frenet_s[-1] else \
                                np.argmax(segment_frenet_s > point[0]) - 1 for point in points])

        t = points[:, 0] - segment_frenet_s[segment_idxs]
        closest_segments = ref_lane_segments[segment_idxs]
        l2 = np.sum((closest_segments[:, 1]-closest_segments[:, 0])**2, axis=-1)

        eps = 1e-8
        l2[l2 < eps] = eps
        proj_points = closest_segments[:, 0] + (t/(l2**0.5))[:, np.newaxis] * (closest_segments[:, 1] - closest_segments[:, 0])
        segment_dirs = closest_segments[:, 1] - closest_segments[:, 0]

        segment_dirs[segment_dirs == 0] = 1e-100
        segment_slopes = segment_dirs[:, 1]/segment_dirs[:, 0]
        segment_perp = -1/segment_slopes
        perp_ints = proj_points[:, 1] - segment_perp*proj_points[:, 0]
        perp_vec = np.stack([proj_points[:, 0], proj_points[:, 1] - perp_ints], axis=-1)
        perp_vec = perp_vec / np.linalg.norm(perp_vec, axis=-1)[:, np.newaxis]
        d = points[:, 1]
        points_pos = proj_points + perp_vec*d[:, np.newaxis]
        points_neg = proj_points - perp_vec*d[:, np.newaxis]

        # See: https://stackoverflow.com/a/1560510/10101616
        AB = segment_dirs
        AX_pos = points_pos - closest_segments[:, 0]
        # AX_neg = points_neg - closest_segments[:, 0]
        indicators_pos = AB[:, 0]*AX_pos[:, 1] - AB[:, 1]*AX_pos[:, 0]
        indicators_pos[indicators_pos < 0] = -1
        indicators_pos[indicators_pos >= 0] = 1

        correct_signs = np.sign(points[:, 1])
        correct_signs[correct_signs == 0] = 1
        xy_out = np.stack([points_pos[i] if indicators_pos[i] == sign else points_neg[i] \
                    for i, sign in enumerate(correct_signs)], axis=0)
        # plt.plot(xy_out[:, 0], xy_out[:, 1], marker='.', label='Recovered')
        return xy_out

    def generate_prediction_dicts(self, batch_dict, output_path=None):
        """

        Args:
            batch_dict:
                pred_scores: (num_center_objects, num_modes)
                pred_trajs: (num_center_objects, num_modes, num_timestamps, 7)

              input_dict:
                center_objects_world: (num_center_objects, 10)
                center_objects_type: (num_center_objects)
                center_objects_id: (num_center_objects)
                center_gt_trajs_src: (num_center_objects, num_timestamps, 10)
        """
        input_dict = batch_dict['input_dict']

        pred_scores = batch_dict['pred_scores']
        pred_trajs = batch_dict['pred_trajs']
        center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)

        num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
        #assert num_feat == 7
        do_rotation = not self.use_frenet

        if do_rotation:
        # if True:
            pred_trajs_world = common_utils.rotate_points_along_z(
                points=pred_trajs.view(num_center_objects, num_modes * num_timestamps, num_feat),
                angle=center_objects_world[:, 6].view(num_center_objects)
            ).view(num_center_objects, num_modes, num_timestamps, num_feat)
        else:
            pred_trajs_world = pred_trajs.view(num_center_objects, num_modes, num_timestamps, num_feat)
        # TODO: agent centric remove?
        pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2]
        pred_trajs_world = pred_trajs_world.cpu()

        # TODO: convert back to xyz from frenet here, knowing each pred is its own center obj; use frenet_lanes field
        if self.use_frenet:
            frenet_lanes = batch_dict['input_dict']['frenet_lanes']
            frenet_lane_lens = batch_dict['input_dict']['frenet_lane_lengths']
            frenet_lanes = [frenet_lane[:frenet_lane_len] for frenet_lane, frenet_lane_len in zip(frenet_lanes, frenet_lane_lens)]
            new_preds = []
            for pred_traj, frenet_lane in zip(pred_trajs_world, frenet_lanes):
                modes = []
                for mode in pred_traj:
                    xy = self.simple_frenet2cartesian(mode.cpu().numpy(), frenet_lane.cpu().numpy())
                    modes.append(torch.from_numpy(xy).to(pred_traj.dtype).to(pred_traj.device))
                modes = torch.stack(modes)
                new_preds.append(modes)
            pred_trajs_world = torch.stack(new_preds)

        pred_dict_list = []
        for obj_idx in range(num_center_objects):
            single_pred_dict = {
                'scenario_id': input_dict['scenario_id'][obj_idx],
                'pred_trajs': pred_trajs_world[obj_idx, :, :, 0:2].numpy(),
                'pred_scores': pred_scores[obj_idx, :].cpu().numpy(),
                'object_id': input_dict['center_objects_id'][obj_idx],
                'object_type': input_dict['center_objects_type'][obj_idx],
                'gt_trajs': input_dict['center_gt_trajs_src'][obj_idx].cpu().numpy(),
                'track_index_to_predict': input_dict['track_index_to_predict'][obj_idx].cpu().numpy()
            }
            pred_dict_list.append(single_pred_dict)

        return pred_dict_list

    def evaluation(self, pred_dicts, output_path=None, eval_method='waymo', model=None, **kwargs):
        if eval_method == 'waymo':
            from .waymo_eval import waymo_evaluation
            try:
                num_modes_for_eval = pred_dicts[0][0]['pred_trajs'].shape[0]
            except:
                num_modes_for_eval = 6
            metric_results, result_format_str = waymo_evaluation(pred_dicts=pred_dicts, num_modes_for_eval=num_modes_for_eval)

            metric_result_str = '\n'
            for key in metric_results:
                metric_results[key] = metric_results[key]
                metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
            metric_result_str += '\n'
            metric_result_str += result_format_str
        else:
            raise NotImplementedError

        return metric_result_str, metric_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    args = parser.parse_args()

    import yaml
    from easydict import EasyDict
    try:
        yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
    except:
        yaml_config = yaml.safe_load(open(args.cfg_file))
    dataset_cfg = EasyDict(yaml_config)



