import argparse
import pdb
import datetime
import os
import pickle as pkl
import logging
from pathlib import Path
import numpy as np
import torch
import hashlib
import sys
import json
import io
import time
import contextlib

from tqdm import tqdm
from copy import deepcopy
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import silhouette_score
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import dtw
from natsort import natsorted
import mdtraj as md
from scipy.interpolate import interp1d
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform
from kneed import KneeLocator
import pandas as pd
import mdtraj.geometry.alignment as align
from mdtraj.utils import ensure_type

# Force using CPU instead of GPU, since calling waymo_eval so often...
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from mtr.datasets.waymo.waymo_eval import waymo_evaluation_explore, waymo_evaluation
from mtr.utils.motion_utils import batch_nms
from mtr.config import cfg, cfg_from_yaml_file
from mtr.utils import common_utils

from utils import (
    plot_static_map_infos, plot_dynamic_map_infos, plot_lanes_by_distance,
    POS_XYZ_IDX, CACHE_DIR, VISUAL_OUT_DIR, POS_XY_IDX, VEL_XY_IDX, HEADING_IDX
)

VISUAL_OUT_SUBDIR = os.path.join(VISUAL_OUT_DIR, f"{os.path.basename(__file__).split('.')[0]}")

def cross_distance(trajs1, trajs2):
    dist_mat = np.zeros((len(trajs1), len(trajs2)), np.float32)
    trajs1 = np.stack(trajs1)    
    trajs1 = trajs1 - trajs1[:, 0, :][:, np.newaxis, :]
    t1 = md.Trajectory(trajs1, topology=None)
    t1._rmsd_traces = np.sum(trajs1**2, axis=-1).sum(axis=-1)
    trajs2 = np.stack(trajs2)    
    trajs2 = trajs2 - trajs2[:, 0, :][:, np.newaxis, :]
    t2 = md.Trajectory(trajs2, topology=None)
    t2._rmsd_traces = np.sum(trajs2**2, axis=-1).sum(axis=-1)

    with tqdm(total=dist_mat.size, desc='Building distance mat') as pbar:
        for i, _ in enumerate(trajs2):
            res = md.rmsd(t1, t2, frame=i, precentered=True)
            dist_mat[:, i] = res
            pbar.update(len(t1))
    return dist_mat


def self_distance(trajs):
    return cross_distance(trajs, trajs)

def get_labels(args):
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    dataset_cfg = cfg.DATA_CONFIG
    dataset_cfg.MODELS = sorted(dataset_cfg.MODELS, key=lambda x: x['ckpt'])
    model_short = hashlib.shake_256(json.dumps(cfg.DATA_CONFIG.MODELS, sort_keys=True).encode()).hexdigest(4)
    label_path = dataset_cfg.DATASET_PATH + '/' + model_short + '.pkl'
    label_path = str(cfg.ROOT_DIR) + '/' + label_path

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag

    if os.path.exists(label_path):
        # Read from cache:
        with open(label_path, 'rb') as f:
            labels = pkl.load(f)
            labels['train'] = natsorted(labels['train'], key=lambda x: x['scenario_id'])
            labels['val'] = natsorted(labels['val'], key=lambda x: x['scenario_id'])
            labels['test'] = natsorted(labels['test'], key=lambda x: x['scenario_id'])
    else:
        raise NotImplementedError('Still need to implement caching stuff')
    return labels, dataset_cfg

def build_cache(base_path, cache_path, num_scenarios=-1, shards=10, num_shards=-1, hist_only=False):
    #splits = ['training', 'validation', 'testing']
    splits = ['validation']

    def load_base(split):
        base = os.path.expanduser(base_path)
        step = 1
        if split == 'training':
            step = 5
            scenarios_base = f'{base}/new_processed_scenarios_training'
            scenarios_meta = f'{base}/new_processed_scenarios_training_infos.pkl'
        elif split == 'validation':
            scenarios_base = f'{base}/new_processed_scenarios_validation'
            scenarios_meta = f'{base}/new_processed_scenarios_val_infos.pkl'
        else:
            scenarios_base = f'{base}/new_processed_scenarios_testing'
            scenarios_meta = f'{base}/new_processed_scenarios_test_infos.pkl'

        start = time.time()
        print(f"Loading {split} Scenario Data...")
        with open(scenarios_meta, 'rb') as f:
            metas = pkl.load(f)[::step]
        inputs = [(f'sample_{x["scenario_id"]}.pkl', f'{scenarios_base}/sample_{x["scenario_id"]}.pkl') for x in metas]
        print(f"Process took {time.time() - start} seconds.")

        start = time.time()
        tot_shards = shards if num_shards == -1 else num_shards

        all_interp_vals = []
        for shard_idx in range(tot_shards):
            print(f"Loading cache {shard_idx} of {tot_shards}")
            file_name = 'interp' if not hist_only else 'interp_hist'
            shard_suffix = f'_shard{shard_idx}_{shards}' if shards > 1 else ''
            interp_vals_filepath = os.path.join(cache_path, f"{split}/frenet/{file_name}{shard_suffix}.npz")
            interp_vals = np.load(interp_vals_filepath, allow_pickle=True)['arr_0']
            all_interp_vals.extend(interp_vals)
        interp_vals = all_interp_vals
        print(f"Loading shards took {time.time() - start} seconds")

        tot_scenarios = len(metas)
        if num_scenarios != -1:
            tot_scenarios = num_scenarios
            metas = metas[:tot_scenarios]
            inputs = inputs[:tot_scenarios]
            interp_vals = interp_vals[:tot_scenarios]

        return metas, inputs, interp_vals
        
    def process_file(input, interp_vals, hist_only=False):
        path = input[1]
        # Load the scenario 
        with open(path, 'rb') as f:
            scenario = pkl.load(f)
        scenario_id = scenario['scenario_id']

        # Trajectory data:
        #    center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid
        track_infos = scenario['track_infos']
        objects_type = track_infos['object_type']

        # interp_val data:
        #    center_x, center_y, center_z, lane_s, lane_d, lane_id, valid
        raw_trajs = track_infos['trajs']
        masks = raw_trajs[:, :, -1] > 0
        # subtract 1 since valid is started directly in mask
        feature_idxs = np.arange(raw_trajs.shape[-1] - 1)

        num_agents, time_steps, dim = interp_vals.shape
        outs = []
        for n in range(num_agents):
            raw_traj = raw_trajs[n]
            interp_val = interp_vals[n]
            last_t = 11 if hist_only else 91
            track_vels = raw_traj[:, feature_idxs[VEL_XY_IDX]]

            breakpoint()

    to_cache = {}
    for split in splits:
        CACHE_SUBDIR = os.path.join(CACHE_DIR, split, 'cluster')
        os.makedirs(CACHE_SUBDIR, exist_ok=True)
        to_cache[split] = {}

        gt_hists = []
        gt_hists_norm = []
        gt_hists_valid = []
        gt_hists_orig = []
        scenario_ids = []
        gt_dists = []
        object_ids = []
        object_types = []

        metas, inputs, interp_vals = load_base(split)
        for _, input, interp_val in tqdm(zip(metas, inputs, interp_vals), 'Processing scenarios', total=len(interp_vals)):
            out = process_file(input, interp_val, hist_only=hist_only)
            
            # def dist_normalize(x):
            #     xyz = x[:11, :3]
            #     dist = np.sqrt(np.sum((xyz[1:] - xyz[:-1])**2, axis=-1))
            #     tot_dist = dist.sum()
            #     # Total distance covered less than 1cm
            #     if tot_dist < 1e-2:
            #         return xyz, tot_dist
            #     tmp = xyz - xyz[0]
            #     tmp = tmp / tot_dist
            #     tmp = tmp + xyz[0]
            #     return tmp, tot_dist
            
            gt_hist_norm, tot_dist = dist_normalize(gt_hists[-1])
            gt_hists_norm.append(gt_hist_norm)
            gt_dists.append(tot_dist)
            scenario_ids.append(scenario_id)
            object_ids.append(object_id)
            object_types.append(object_type)

        object_types = np.stack(object_types)
        scenario_ids = np.stack(scenario_ids)
        object_ids = np.stack(object_ids)
        object_types = np.stack(object_types)
        gt_hists = np.stack(gt_hists)
        gt_hists_orig = np.stack(gt_hists_orig)
        gt_hists_valid = np.stack(gt_hists_valid)
        gt_hists_norm = np.stack(gt_hists_norm)
        gt_dists = np.stack(gt_dists)

        # Now, we would like to impose thresholds on total distance covered + object type
        for object_type in np.unique(object_types):
            object_idxs = np.arange(len(object_types))[object_types == object_type]
            object_dists = gt_dists[object_idxs]

            # Stationary should be 0.25 m/s, i.e. 0.025 per timestep
            stationary = 0.25/10
            n_quartiles = 1
            quartiles = np.linspace(0, 100, n_quartiles+1)
            quartiles = [np.percentile(object_dists, quartile) for quartile in quartiles][1:]
            cutoffs = [stationary, *quartiles]
            cutoffs = [(-stationary, cutoffs[0]), *zip(cutoffs, cutoffs[1:])]
            cutoff_labels = ['stationary', *[f'speed{i}' for i in range(1, 1+len(quartiles))]]
            dist_infos = {}
            for cutoff, label in zip(cutoffs, cutoff_labels):
                cutoff_idxs = object_idxs[(object_dists > cutoff[0]) & (object_dists <= cutoff[1])]
                info = {}
                info['gt_hists'] = gt_hists[cutoff_idxs]
                info['scenario_ids'] = scenario_ids[cutoff_idxs]
                info['object_ids'] = object_ids[cutoff_idxs]
                info['object_types'] = object_types[cutoff_idxs]
                info['gt_hists_orig'] = gt_hists_orig[cutoff_idxs]
                info['gt_hists_valid'] = gt_hists_valid[cutoff_idxs]
                info['gt_hists_norm'] = gt_hists_norm[cutoff_idxs]
                info['gt_dists'] = gt_dists[cutoff_idxs]
                if split == 'train':
                    dist_mat = self_distance(info['gt_hists_norm'])
                    info['dists'] = dist_mat
                dist_infos[label] = info

            to_cache[split][object_type] = dist_infos
    return to_cache


def perform_clustering(train_info):
    for object_type, object_info in train_info.items():
        for label, info in object_info.items():
            dists = info['dists']

            # Takes up to ~120 seconds, or for 25k takes around 25 seconds
            num = 40000
            dists = dists[:num, :num]
            min_samples = 10
            n_clusters = 20
            print('Clustering', object_type, label, 'n =', len(dists), f'(from {len(info["gt_hists"])})')
            if len(dists) < min_samples*n_clusters:
                info['cluster_info'] = {'labels': np.zeros((len(dists))), 'idxs': np.arange(len(dists)), 'linkage': None}
                continue

            reduced_distances = squareform(dists, checks=False)
            linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method='ward')
            labels = scipy.cluster.hierarchy.fcluster(linkage, 1, criterion='distance').squeeze()
            unique_labels, unique_labels_cnt = np.unique(labels, return_counts=True)
            labels_order = np.argsort(unique_labels_cnt)[::-1]
            new_labels = np.zeros_like(labels)
            for new_label, label_val in enumerate(unique_labels[labels_order]):
                new_labels[labels == label_val] = new_label
            labels = new_labels
            info['cluster_info'] = {'labels': labels, 'idxs': np.arange(len(dists)), 'linkage': linkage} 

def assign_full_cluster(to_cache):
    for split, split_info in to_cache.items():
        assert len(set([frozenset(x.keys()) for x in split_info.values()])) == 1, 'Mismatched speed splits'
        for object_type, object_info in split_info.items():
            for label, info in object_info.items():
                print(f'Assigning full clusters for {split} {object_type} {label}')
                hists = info['gt_hists_norm']
                cluster_info = to_cache['train'][object_type][label]['cluster_info']
                cluster_packed = (cluster_info['idxs'] == np.array(range(0, len(cluster_info['idxs'])))).all()
                if cluster_packed:
                    cluster_trajs = to_cache['train'][object_type][label]['gt_hists_norm'][:len(cluster_info['idxs'])]
                else:
                    cluster_trajs = to_cache['train'][object_type][label]['gt_hists_norm'][cluster_info['idxs']]
                if split == 'train':
                    if cluster_packed:
                        dists = to_cache['train'][object_type][label]['dists'][:, :len(cluster_info['idxs'])]
                    else:
                        dists = to_cache['train'][object_type][label]['dists'][:, cluster_info['idxs']]
                else:
                    dists = cross_distance(hists, cluster_trajs)

                if cluster_packed:
                    ref_dist = to_cache['train'][object_type][label]['dists'][:len(cluster_info['idxs']), :len(cluster_info['idxs'])] 
                else:
                    ref_dist = to_cache['train'][object_type][label]['dists'][cluster_info['idxs']][:, cluster_info['idxs']]
                print('Predicting labels')
                KN = KNeighborsClassifier(n_neighbors=min(5, len(ref_dist)), metric='precomputed').fit(ref_dist, cluster_info['labels'])
                new_labels = KN.predict(dists)
                info['predicted_labels'] = new_labels
                cluster_idxs = np.unique(new_labels)

                representative_idxs = []
                for cluster_idx in tqdm(cluster_idxs, 'Finding representative idxs'):
                    sub_dist = dists[new_labels == cluster_idx][:, cluster_info['labels'] == cluster_idx]
                    representative_idx = sub_dist.mean(axis=1).argmin()
                    representative_idxs.append(representative_idx)
                info['representative_idxs'] = representative_idxs

def visualize_clusters(to_cache):
    for split, split_info in to_cache.items():
        for object_type, object_info in split_info.items():
            for label, info in object_info.items():
                object_labels = info['predicted_labels']
                object_trajs = info['gt_hists_norm']

                cluster_sizes = np.unique(object_labels, return_counts=True)[1]
                # Sort high to low
                clusters_sorted = np.argsort(cluster_sizes)[::-1]
                cluster_vis_path = f'cluster_vis/{split}/{object_type.split("_")[1].lower()}/{label}'
                os.makedirs(cluster_vis_path, exist_ok=True)

                cluster_speeds = []
                representative_elements = []
                for cluster_idx in tqdm(clusters_sorted, f'processing cluster for {split} {object_type} {label}'):
                    cluster_size = cluster_sizes[cluster_idx]
                    # if cluster_size < 10:
                    #     continue
                    cluster_trajs = object_trajs[object_labels == cluster_idx]
                    cluster_speed = np.zeros(cluster_trajs.shape[:-1])
                    cluster_speed[:, 1:] = np.sqrt(np.sum((cluster_trajs[:, 1:] - cluster_trajs[:, :-1])**2, axis=-1))
                    cluster_speed = cluster_speed[:, 1:].mean()
                    cluster_speeds.append(cluster_speed)

                    representative_idx = info['representative_idxs'][cluster_idx]

                    t1 = md.Trajectory(cluster_trajs, topology=None).center_coordinates()
                    #aligned_cluster_trajs = t1.superpose(t1[0]).xyz
                    # Figure out alignment manually
                    aligned_cluster_trajs = t1.xyz
                    aligned_cluster_trajs = aligned_cluster_trajs - aligned_cluster_trajs[representative_idx, 0, :]
                    target = aligned_cluster_trajs[representative_idx]
                    avg_angle = np.arctan2((target[1:] - target[:-1])[:, 0], (target[1:] - target[:-1])[:, 1])[0]
                    target_rot = Rotation.from_euler('z', avg_angle - np.pi/2, degrees=False).as_matrix()
                    target = (target_rot@target.transpose(1, 0)).transpose(1, 0)
                    representative_elements.append(target)
                    plt.clf()
                    for traj in aligned_cluster_trajs:
                        mobile = traj
                        # trans, rot = compute_translation_and_rotation_start(mobile, target)
                        # mobile_centered = (rot@(mobile - target).transpose(1, 0)).transpose(1, 0)
                        mobile_centered = align.transform(mobile, target)
                        plt.plot(mobile_centered[:, 0], mobile_centered[:, 1], marker='.', color='b', alpha=0.01)
                    plt.plot(target[:, 0], target[:, 1], marker='.', color='r', alpha=1)
                    plt.xlim(-1, 1)
                    plt.ylim(-1, 1)
                    plt.title(f'Cluster {cluster_idx}, n={cluster_size}')
                    plt.savefig(f'{cluster_vis_path}/cluster_{cluster_idx}.png', dpi=200)
                
                if len(representative_elements):
                    target = representative_elements[0]
                    plt.clf()
                    for traj in representative_elements:
                        mobile = traj
                        mobile_centered = align.transform(mobile, target)
                        plt.plot(mobile_centered[:, 0], mobile_centered[:, 1], marker='.', alpha=0.25)
                    plt.xlim(-1, 1)
                    plt.ylim(-1, 1)
                    plt.title('Representative Elements of Each Cluster')
                    plt.savefig(f'{cluster_vis_path}/representatives.png', dpi=200)

def assess_labels(labels, to_cache, dataset_cfg):
    # Look at val set to start
    splits = labels.keys()
    all_results = {split: [] for split in splits}
    for model in tqdm(dataset_cfg.MODELS, desc='Loading results...'):
        for split in splits:
            res_path = str(cfg.ROOT_DIR) + '/' + model[f'{split}_results']
            with open(res_path, 'rb') as f:
                results = pkl.load(f)
            results = natsorted(results, key=lambda x: x['scenario_id'])
            all_results[split].append(results)
    labels_dict = {}
    for split in splits:
        labels_split = labels[split]
        labels_dict[split] = {}
        for label in labels_split:
            scenario_id = label['scenario_id']
            object_id = label['object_id']
            if scenario_id in labels_dict[split]:
                labels_dict[split][scenario_id][object_id] = label
            else:
                labels_dict[split][scenario_id] = {object_id: label}
    cluster_dict = {}
    for split, split_info in to_cache.items():
        cluster_dict[split] = {}
        for object_type, object_info in split_info.items():
            for label, info in object_info.items():
                for scenario_id, object_id, predicted_cluster in \
                      zip(info['scenario_ids'], info['object_ids'], info['predicted_labels']):
                    inner_dict = {'object_type': object_type, 'predicted_cluster': predicted_cluster, 'cluster_label': label}
                    if scenario_id in cluster_dict[split]:
                        cluster_dict[split][scenario_id][object_id] = inner_dict
                    else:
                        cluster_dict[split][scenario_id] = {object_id: inner_dict}
    

    output = {}
    for split in labels.keys():
        output[split] = {}
        all_results_split = all_results[split]
        for paired_result in tqdm(zip(*all_results_split), 'Processing scenarios', total=len(all_results_split[0])):
            assert len(set([result['scenario_id'] for result in paired_result])) == 1, 'Mismatched scenario_id'
            assert len(set([result['object_id'] for result in paired_result])) == 1, 'Mismatched object_id'
            scenario_id = paired_result[0]['scenario_id']
            object_id = paired_result[0]['object_id']
            label_res = labels_dict[split][scenario_id][object_id]
            cluster_res = cluster_dict[split][scenario_id][object_id]
            
            object_type = cluster_res['object_type']
            cluster_label = cluster_res['cluster_label']
            predicted_cluster = cluster_res['predicted_cluster']
            if object_type not in output[split]:
                output[split][object_type] = {}
            if cluster_label not in output[split][object_type]:
                output[split][object_type][cluster_label] = []
            
            inner_dict = {
                'scenario_id': scenario_id,
                'object_id': object_id,
                'predicted_cluster': predicted_cluster
            }
            for i, val in enumerate(label_res['mAPs']):
                inner_dict[f'mAP_{i}'] = val
            output[split][object_type][cluster_label].append(inner_dict)
    
    n_models = len(dataset_cfg.MODELS)
    cols = [f'mAP_{i}' for i in range(n_models)]
    output_raw = deepcopy(output)
    for split, split_info in output.items():
        for object_type, object_info in split_info.items():
            for label, label_info in object_info.items():
                inner_data = pd.DataFrame(label_info)
                output[split][object_type][label] = inner_data
                output_raw[split][object_type][label] = inner_data.groupby('predicted_cluster').mean()[cols].to_numpy()

    train_res = output['train']['TYPE_VEHICLE']['speed1'].groupby('predicted_cluster').mean()[cols].to_numpy()
    val_res = output['val']['TYPE_VEHICLE']['speed1'].groupby('predicted_cluster').mean()[cols].to_numpy()
    test_res = output['test']['TYPE_VEHICLE']['speed1'].groupby('predicted_cluster').mean()[cols].to_numpy()

    print(train_res.argmax(axis=1))
    print(val_res.argmax(axis=1))
    print(test_res.argmax(axis=1))

    def get_str(results):
        metric_results, result_format_str  = waymo_evaluation(results, num_modes_for_eval=len(results[0]['pred_trajs']))
        metric_result_str = '\n'
        for key in metric_results:
            metric_results[key] = metric_results[key]
            metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
        metric_result_str += '\n'
        metric_result_str += result_format_str
        return metric_result_str
    
    def filter_str(str):
        return '\n'.join(str.split('\n')[-7:-1])

    for split in labels.keys():
        all_results_split = all_results[split]
        greedy_results = []
        print(f'****** {split.upper()} ******')
        for paired_result in tqdm(zip(*all_results_split), 'Processing scenarios', total=len(all_results_split[0])):
            scenario_id = paired_result[0]['scenario_id']
            object_id = paired_result[0]['object_id']
            cluster_res = cluster_dict[split][scenario_id][object_id]
            
            object_type = cluster_res['object_type']
            cluster_label = cluster_res['cluster_label']
            predicted_cluster = int(cluster_res['predicted_cluster'])
            greedy_idx = output_raw['train'][object_type][cluster_label][predicted_cluster].argmax()
            greedy_results.append(paired_result[greedy_idx])
        print(filter_str(get_str(greedy_results)))

    import pdb; pdb.set_trace()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='~/monet_shared/shared/mtr_process')
    parser.add_argument('--cache_path', type=str, default='/av_shared_ssd/scenario_id/cache')
    parser.add_argument('--num_scenarios', type=int, default=-1)
    parser.add_argument('--cfg_file', default='../cfgs/meta/mtr+20p_mini.yaml', help='Which ensemble config to use')
    parser.add_argument('--extra_tag', default='default')
    parser.add_argument('--split', default='training', choices=['training', 'validation', 'testing'])
    parser.add_argument('--cache_file', default='../../data/clustering/initial.pkl', help='Output file')
    parser.add_argument('--shards', type=int, default=10)
    parser.add_argument('--num_shards', type=int, default=-1)
    parser.add_argument('--hist_only', action='store_true')
    parser.add_argument('--recache', action='store_true', help='Ignore/rebuild cache')
    args = parser.parse_args()

    os.makedirs(VISUAL_OUT_SUBDIR, exist_ok=True)

    if not args.recache and os.path.exists(args.cache_file):
        print('Loading cache...')
        with open(args.cache_file, 'rb') as f:
            to_cache = pkl.load(f)

        ############# PERFORM CLUSTERING ##############
        # perform_clustering(to_cache['train'])
        # with open(args.cache_file, 'wb') as f:
        #     pkl.dump(to_cache, f)

        ################ ASSIGNING CLUSTERS TO FULL TRAIN/VAL/TEST ###################
        # assign_full_cluster(to_cache)
        # with open(args.cache_file, 'wb') as f:
        #     pkl.dump(to_cache, f)

        ####################### VISUALIZING CLUSTERS FOR TRAINING #################
        # visualize_clusters(to_cache)

        ################# ASSESS SPECIFIED ENSEMBLE #####################
        # labels, dataset_cfg = get_labels(args)
        # assess_labels(labels, to_cache, dataset_cfg)
        sys.exit(0)


    #labels, dataset_cfg = get_labels(args)

    to_cache = build_cache(args.base_path, args.cache_path, args.num_scenarios,
                           args.shards, args.num_shards, args.hist_only)
    with open(args.cache_file, 'wb') as f:
        pkl.dump(to_cache, f)
    import pdb; pdb.set_trace()

    perform_clustering(to_cache['train'])
    with open(args.cache_file, 'wb') as f:
        pkl.dump(to_cache, f)
    import pdb; pdb.set_trace()

    assign_full_cluster(to_cache)
    with open(args.cache_file, 'wb') as f:
        pkl.dump(to_cache, f)
    import pdb; pdb.set_trace()

    visualize_clusters(to_cache)

    # TODO for real:
    # - think about how to handle missing data points...is interpolating really the move? 
    #   - Analyze how they occur...is it late start? Early end? Missing middle?
    # - should we care about road/lane topology? As in, Frenet coordinates rather than xyz -> maybe...
    # - consider V2V stuff, as in the journal article (read Appendix C and rest of paper...)
    #   - just do clustering on the L1 offsets at each time step
    import pdb; pdb.set_trace()

