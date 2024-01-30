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
import contextlib
from joblib import Parallel, delayed

from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
import dtw
from natsort import natsorted
from shapely import LineString

# Force using CPU instead of GPU, since calling waymo_eval so often...
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from mtr.datasets.waymo.waymo_eval import waymo_evaluation_extra, waymo_evaluation
from mtr.utils.motion_utils import batch_nms
from mtr.config import cfg, cfg_from_yaml_file
from mtr.utils import common_utils

import contextlib
import joblib
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def process_scenario(scenario_result, scenario_path, collision_thresholds=[0.1, 0.25, 0.5, 1.0], use_gt=False):
    assert os.path.exists(scenario_path), f'Scenario path not found: {scenario_path}' 
    # scenario_result is a list of all the center objects predicts, with dict info for like object id, etc.
    # Keys: scenario_id, pred_trajs, pred_scores, object_id, object_type, gt_trajs, track_index_to_predict
    # Point will be to compare *each* predicted future center with the GT future of everybody else in the scene

    with open(scenario_path, 'rb') as f:
        scenario = pkl.load(f)
    
    # First we check for collisions, for each of the predicted traj
    out = {}
    for thresh in collision_thresholds:
        out[f'collisions_{thresh} - VEHICLE'] = 0
        out[f'collisions_{thresh} - PEDESTRIAN'] = 0
        out[f'collisions_{thresh} - CYCLIST'] = 0
    out['trajs - VEHICLE'] = 0
    out['trajs - PEDESTRIAN'] = 0
    out['trajs - CYCLIST'] = 0

    for agent_result in scenario_result:
        # 6 x 80 x 2
        pred_traj = agent_result['pred_trajs']
        assert pred_traj.shape == (6, 80, 2), 'Unexpected number of modes provided'
        # Need to care about unknown future values for some agents
        other_trajs = scenario['track_infos']['trajs'][:, 11:, :2]
        other_trajs_valid = scenario['track_infos']['trajs'][:, 11:, -1]
        if not use_gt:
            # Take out ego agent
            other_trajs = np.stack([x for x, object_id in zip(other_trajs, scenario['track_infos']['object_id']) \
                                    if object_id != agent_result['object_id']])
            other_trajs_valid = np.stack([x for x, object_id in zip(other_trajs_valid, scenario['track_infos']['object_id']) \
                                    if object_id != agent_result['object_id']]).astype(bool)
        else:
            # Shape should be 1 x 80 x 2
            pred_traj = np.stack([x for x, object_id in zip(other_trajs, scenario['track_infos']['object_id'])  if object_id == agent_result['object_id']])
            # Shape should be 1 x 80
            pred_traj_valid = np.stack([x for x, object_id in zip(other_trajs_valid, scenario['track_infos']['object_id']) \
                                    if object_id == agent_result['object_id']]).astype(bool)
            # Take out ego agent
            other_trajs = np.stack([x for x, object_id in zip(other_trajs, scenario['track_infos']['object_id']) \
                                    if object_id != agent_result['object_id']])
            other_trajs_valid = np.stack([x for x, object_id in zip(other_trajs_valid, scenario['track_infos']['object_id']) \
                                    if object_id != agent_result['object_id']]).astype(bool)
            other_trajs_valid &= pred_traj_valid
            
        # e.g. 30 -> 29
        assert len(other_trajs_valid) == len(other_trajs) and \
                len(other_trajs) == len(scenario['track_infos']['trajs']) - 1, 'Could not find object id'
        # e.g. 6 x 29 x 80
        dists = np.linalg.norm(other_trajs - pred_traj[:, np.newaxis, :, :], axis=-1)
        n_pred = dists.shape[0]

        obj_type = agent_result['object_type'].split('_')[-1]
        #valid_dists = dists[:, other_trajs_valid]
        valid_dists = [dists[:, i, traj_valid] for i, traj_valid in enumerate(other_trajs_valid)]
        segment_overlap_collisions = []
        for other_traj, other_traj_mask, valid_dist in zip(other_trajs, other_trajs_valid, valid_dists):
            other_traj = other_traj[other_traj_mask]
            # i.e. would only miss if moving faster than 100 m/s in same directions, 50 m/s in opposite
            if len(other_traj) < 2 or valid_dist.min() > 10:
                segment_overlap_collisions.append([0] * n_pred)
                continue
            traj_overlaps = []

            for traj, traj_dist in zip(pred_traj, valid_dist):
                traj = traj[other_traj_mask]
                if traj_dist.min() > 10:
                    traj_overlaps.append(0)
                    continue
                segment_dist = [min(traj_dist[i], traj_dist[i+1]) for i in range(len(traj_dist) - 1)]
                segments_j = np.stack([other_traj[:-1], other_traj[1:]], axis=1)
                segments_j = [x for x, dist in zip(segments_j, segment_dist) if dist <= 10]
                if not len(segments_j):
                    traj_overlaps.append(0)
                    continue
                segments_j = np.array([LineString(x) for x in segments_j])

                segments_i = np.stack([traj[:-1], traj[1:]], axis=1)
                segments_i = [x for x, dist in zip(segments_i, segment_dist) if dist <= 10]
                segments_i = np.array([LineString(x) for x in segments_i])

                overlaps = (np.array([False] + [x.intersects(y) for x, y in zip(segments_i, segments_j)]))
                if overlaps.any():
                    traj_overlaps.append(1)
                else:
                    traj_overlaps.append(0)
            segment_overlap_collisions.append(traj_overlaps)

        for thresh in collision_thresholds:
            # Each pair of (agent_traj, other_traj) can count for at most one collision, hence the any() over time
            # Do not want to double count with overlap collisions from above
            #old_collisions = np.sum([np.any(valid_dist < thresh, axis=-1) for valid_dist in valid_dists])
            collisions = np.sum([np.sum([max(x, y) for x, y in zip(np.any(valid_dist < thresh, axis=-1), overlaps)]) for valid_dist, overlaps in zip(valid_dists, segment_overlap_collisions)])
            out[f'collisions_{thresh} - {obj_type}'] += collisions
        out[f'trajs - {obj_type}'] += n_pred
    
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', default='cfgs/mini/vrnn+20p_64_a_test.yaml', help='Which config to use')
    parser.add_argument('--extra_tag', default='default')
    parser.add_argument('--epoch', default='epoch_best')
    parser.add_argument('--gt', action='store_true', help='Whether or not to use GT instead of predicted traj')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--nproc', default=20, type=int)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    dataset_cfg = cfg.DATA_CONFIG

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    os.makedirs(output_dir, exist_ok=True)

    test_result = os.path.join(output_dir, 'eval', args.epoch, 'default', 'result.pkl')
    val_result = os.path.join(output_dir, 'eval', args.epoch, 'default', 'result_val.pkl')
    assert os.path.exists(val_result) and os.path.exists(test_result), 'Val and test results must exist'

    if args.gt:
        log_file = output_dir / ('log_gt_metrics_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    else:
        log_file = output_dir / ('log_metrics_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start logging**********************')

    splits = ['val', 'test']
    sample_dirs = []
    results = []
    for split, result_path in zip(splits, [val_result, test_result]):
        #meta_file = os.path.join('..', dataset_cfg.DATA_ROOT, dataset_cfg.INFO_FILE[split])
        sample_dir = os.path.join('..', dataset_cfg.DATA_ROOT, dataset_cfg.SPLIT_DIR[split])
        sample_dirs.append(sample_dir)
        logger.info(f'Loading {split} result from {result_path}')
        with open(result_path, 'rb') as f:
            result = pkl.load(f)
        results.append(result)
    
    # TODO in this script:
    # 1. Invoke waymo eval
    #    a. average the cyclist and pedestrian values as needed for our table
    # 2. Compute more advanced stats, like collision rates at various thresholds
    # 3. Compute out the increase/decrease in percentages for each metric

    log_outs = []
    for split, split_results, sample_dir in zip(['val', 'test'], results, sample_dirs):

        logger.info(f'Processing {split}...')
        scenario_ids = set([x['scenario_id'] for x in split_results])
        scenario_results = {}
        for result in split_results:
            scenario_id = result['scenario_id']
            if scenario_id in scenario_results:
                scenario_results[scenario_id].append(result)
            else:
                scenario_results[scenario_id] = [result]

        msg = 'Processing scenarios'
        if args.parallel:
            from joblib import Parallel, delayed
            with tqdm_joblib(tqdm(desc=msg, total=len(scenario_results))) as pbar:
                all_outs = Parallel(n_jobs=args.nproc, batch_size=4)(delayed(process_scenario)(
                    scenario_result, os.path.join(sample_dir, f'sample_{scenario_id}.pkl'), use_gt=args.gt)
                                            for scenario_id, scenario_result in scenario_results.items())
        else:
            all_outs = []
            for scenario_id, scenario_result in tqdm(scenario_results.items(), msg, total=len(scenario_results)):
                out = process_scenario(scenario_result, os.path.join(sample_dir, f'sample_{scenario_id}.pkl'), use_gt=args.gt)
                all_outs.append(out)
        collision_keys = [x for x in all_outs[0].keys() if 'collision' in x]
        collision_rates = {}
        for key in collision_keys:
            obj_type = key.split(' - ')[-1]
            all_cols = np.sum([x[key] for x in all_outs])
            tot_trajs = np.sum([x[f'trajs - {obj_type}'] for x in all_outs])
            collision_rates[key] = all_cols/tot_trajs
        
        metric_results, result_format_str = waymo_evaluation_extra(split_results, extra_val=collision_rates)

        metric_result_str = '\n'
        for key in metric_results:
            metric_results[key] = metric_results[key]
            metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
        metric_result_str += '\n'
        metric_result_str += result_format_str

        log_outs.append(metric_result_str)
        tmp_res ='\n'.join(metric_result_str.split('\n')[-7:-1]) 
        logger.info(f'Cur res: \n\n{tmp_res}\n')

    log_outs = ['\n'.join(x.split('\n')[-7:-1]) for x in log_outs]
    logger.info('\n')
    logger.info(f'Validation results: \n\n{log_outs[0]}\n')

    val = log_outs[0].split('\n')
    test = log_outs[1].split('\n')
    assert len(val) == len(test), 'Must match len'
    # Skip header row
    for i in range(1, len(val)):
        line_val = (' '.join(val[i].split())).split()
        line_test = (' '.join(test[i].split())).split()
        assert len(line_val) == len(line_test), 'Must match len'
        
        for j in range(len(line_val)):
            if not len(line_val[j].split(',')) == 2:
                continue
            try:
                float_value_x = float(line_val[j].split(',')[0])
                float_value_y = float(line_test[j].split(',')[0])
                if float_value_y > float_value_x:
                    append = f'(+{(float_value_y/float_value_x - 1)*100:.2f}%)'
                else: 
                    append = f'(-{(1 - float_value_y/float_value_x)*100:.2f}%)'
                line_test[j] = f'{line_test[j].split(",")[0]} {append},'
                #result_format_str = ' '.join([x.rjust(16) for items in result_format_list for x in items])
            except Exception as e:
                pass
        test[i] = ' '.join([x.rjust(20) for x in line_test])
    log_outs[1] = '\n'.join(test)
    logger.info(f'Test results: \n\n{log_outs[1]}\n')

    
