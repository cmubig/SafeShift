import argparse
import pdb
import os
import pickle as pkl
import pandas as pd
import pingouin as pg
from pathlib import Path
import hashlib
import sys
import json
import io

from tqdm import tqdm
from matplotlib import pyplot as plt
from natsort import natsorted
from operator import itemgetter
import uuid
import time

from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt

import numpy as np
import itertools


from mtr.config import cfg, cfg_from_yaml_file

base = os.path.expanduser('~/monet_shared/shared/mtr_process')

# Uses the "new" splits, from resplit.py; this way, test is labeled as well
train_base = f'{base}/new_processed_scenarios_training'
val_base = f'{base}/new_processed_scenarios_validation'
test_base = f'{base}/new_processed_scenarios_testing'
train_meta = f'{base}/new_processed_scenarios_training_infos.pkl'
val_meta = f'{base}/new_processed_scenarios_val_infos.pkl'
test_meta = f'{base}/new_processed_scenarios_test_infos.pkl'

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', default='../cfgs/meta/mtr+20p_initial.yaml', help='Which ensemble config to use')
    parser.add_argument('--extra_tag', default='default')
    parser.add_argument('--cache_file', default='../../data/clustering/v2v.pkl')
    parser.add_argument('--recache', action='store_true', help='Ignore/rebuild cache')
    args = parser.parse_args()
    # Load meta pickle things; takes ~30s
    with open(train_meta, 'rb') as f:
        train_metas = pkl.load(f)
    with open(val_meta, 'rb') as f:
        val_metas = pkl.load(f)
    with open(test_meta, 'rb') as f:
        test_metas = pkl.load(f)
    
    labels, dataset_cfg = get_labels(args)
    train_metas = train_metas[::5]
    for train_meta in tqdm(train_metas, 'Processing scenarios...', total=len(train_metas)):
        scenario_path = f'{train_base}/sample_{train_meta["scenario_id"]}.pkl'
        with open(scenario_path, 'rb') as f:
            scenario = pkl.load(f)
        sdc_traj = scenario['track_infos']['trajs'][scenario['sdc_track_index']]
        to_predict_trajs = scenario['track_infos']['trajs'][scenario['tracks_to_predict']['track_index']]


        def interpolate(x):
            import pdb; pdb.set_trace()
            hist_valid = (x[:11, -1] == 1)
            hist = x[:11, :9]
            hist[~hist_valid] = np.nan
            if sum(hist_valid) == 0:
                raise NotImplementedError('Unsure how to handle no valid points in hist')
            elif sum(hist_valid) == 1:
                return pd.DataFrame(hist).fillna(method='ffill').fillna(method='bfill').to_numpy()
            else:
                return pd.DataFrame(hist).interpolate(method='slinear', limit_direction='both', fill_value='extrapolate').to_numpy()
            
        def dist_normalize(x):
            xyz = x[:11, :3]
            dist = np.sqrt(np.sum((xyz[1:] - xyz[:-1])**2, axis=-1))
            tot_dist = dist.sum()
            # Total distance covered less than 1cm
            if tot_dist < 1e-2:
                return xyz, tot_dist
            tmp = xyz - xyz[0]
            tmp = tmp / tot_dist
            tmp = tmp + xyz[0]
            return tmp, tot_dist

        trajs = scenario['track_info']['trajs']
        hists_orig = trajs[:, :11]
        hists_valid = trajs[:, :11, -1]
        hists = interpolate(hists_orig)
        gt_hist_norm, tot_dist = dist_normalize(hists)
        
        import pdb; pdb.set_trace()
    