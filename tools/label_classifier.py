import argparse
import pdb
import os
import pickle as pkl
import logging
import numpy as np
import torch
import hashlib
import sys
import json
import io
import contextlib
from contextlib import contextmanager

from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
from pathlib import Path
import dtw
from natsort import natsorted
from joblib import Parallel, delayed

# Force using CPU instead of GPU, since calling waymo_eval so often...
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from mtr.datasets.waymo.waymo_eval import waymo_evaluation_explore, waymo_evaluation
from mtr.utils.motion_utils import batch_nms
from mtr.config import cfg, cfg_from_yaml_file
from mtr.utils import common_utils

# Use GT Trajs to select the best ones, provide an upper bound
def get_best(paired_result, random_state=None):
    pred_scores = []
    pred_trajs = []
    mAPs = []
    possible_keys = ['mAP - VEHICLE', 'mAP - PEDESTRIAN', 'mAP - CYCLIST']
    for result in paired_result:
        avg_results = waymo_evaluation_explore([result], num_modes_for_eval=6)
        mAP = 0
        n_val = 0
        for key in possible_keys:
            if avg_results[key] != -1:
                mAP = avg_results[key]
                n_val += 1
        assert n_val == 1, 'Key not found'
        mAPs.append(mAP)
        pred_scores.append(result['pred_scores'])
        pred_trajs.append(result['pred_trajs'])
    best_indices_mAP = np.where(mAPs == np.max(mAPs))[0]
    assert len(best_indices_mAP) >= 1, 'Must match at least one mAP val'


    all_trajs = np.concatenate(pred_trajs)
    # [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
    gt_xy = paired_result[0]['gt_trajs'][-80:, :2]
    gt_valid = paired_result[0]['gt_trajs'][-80:, 9]

    gt_xy = gt_xy[gt_valid == 1]
    pred_xy = all_trajs[:, gt_valid == 1]
    try:
        ade = np.mean(np.sqrt(np.sum((pred_xy - gt_xy)**2, axis=-1)), axis=-1)
        fde = np.sqrt(np.sum((pred_xy - gt_xy)**2, axis=-1))[:, -1]
    except Exception as e:
        ade = np.zeros(ade.shape).astype(np.float32)
        fde = np.zeros(ade.shape).astype(np.float32)

    # Break ties randomly? Or break ties by ADE...
    best_min_ade = np.inf
    best_idx_map = -1
    for possible_idx in best_indices_mAP:
        min_ade = ade[possible_idx*6:(possible_idx+1)*6].min()
        if min_ade < best_min_ade:
            best_min_ade = min_ade
            best_idx_map = possible_idx
        

    best_idx_ade = ade.argmin() // 6
    best_idx_fde = fde.argmin() // 6
    ret =  {
        'mAP': {'idx': best_idx_map, 'val': np.max(mAPs)}, 
        'ade': {'idx': best_idx_ade, 'val': np.min(ade)}, 
        'fde': {'idx': best_idx_fde, 'val': np.min(fde)},
        'mAPs': mAPs,
        'ades': ade,
        'fdes': fde
    }
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', default='cfgs/meta/mtr+20p_initial.yaml', help='Which ensemble config to use')
    parser.add_argument('--extra_tag', default='default')
    args = parser.parse_args()
    
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    dataset_cfg = cfg.DATA_CONFIG
    dataset_cfg.MODELS = sorted(dataset_cfg.MODELS, key=lambda x: x['ckpt'])
    model_short = hashlib.shake_256(json.dumps(cfg.DATA_CONFIG.MODELS, sort_keys=True).encode()).hexdigest(4)
    label_path = dataset_cfg.DATASET_PATH + '/' + model_short + '.pkl'
    # label_path is the out_directory
    label_path = str(cfg.ROOT_DIR) + '/' + label_path


    # Lol, the epoch_20 comes from a bug in test.py, where since the ckpt is best_ckpt.pth, the only number
    # found is the 20 from 20p
    base_labels = ['train', 'val', 'test']

    to_cache = {}
    random_state = np.random.RandomState(42)
    for base_label in base_labels:
        results_paths = [cfg.ROOT_DIR / model[f'{base_label}_results'] for model in dataset_cfg.MODELS]
        all_results = []
        for results_path in results_paths:
            assert os.path.exists(results_path), f'Missing path: {results_path}'
            with open(results_path, 'rb') as f:
                results = pkl.load(f)
            all_results.append(results)
    
        def process_paired_result(paired_result):
            assert len(set([result['scenario_id'] for result in paired_result])) == 1, 'Mismatched scenario_id'
            assert len(set([result['object_id'] for result in paired_result])) == 1, 'Mismatched object_id'
            result = get_best(paired_result, random_state=random_state)
            result['scenario_id'] = paired_result[0]['scenario_id']
            result['object_id'] = paired_result[0]['object_id']
            return result

        results = Parallel(n_jobs=os.cpu_count())(delayed(process_paired_result)(paired_result) \
                                     for paired_result in tqdm(zip(*all_results), 'Processing scenarios', total=len(all_results[0])))
        to_cache[base_label] = results

    with open(label_path, 'wb') as f:
        pkl.dump(to_cache, f)
