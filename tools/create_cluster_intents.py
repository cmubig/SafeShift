import argparse
import pdb
import os
import pickle as pkl
import pandas as pd
import pingouin as pg

from tqdm import tqdm
from matplotlib import pyplot as plt
from natsort import natsorted
from operator import itemgetter
from sklearn.cluster import KMeans, DBSCAN
import uuid
import time

from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt

import numpy as np
import itertools


base = os.path.expanduser('~/waymo/mtr_process')

# Uses the "new" splits, from resplit.py; this way, test is labeled as well
train_base = f'{base}/joint_original'
val_base = f'{base}/joint_original'
test_base = f'{base}/joint_original'
train_meta = f'{base}/new_processed_scenarios_training_infos.pkl'
val_meta = f'{base}/new_processed_scenarios_val_infos.pkl'
test_meta = f'{base}/new_processed_scenarios_test_infos.pkl'

cluster_out_path = '../data/waymo/new_cluster_%i_center_dict.pkl'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=64, help='how many clusters?')
    parser.add_argument('--inputs', type=str, default='new', help='which input',
                        choices=['new', 'score_gt_80', 'score_fe_80', 'score_combined_80', 
                                 'score_asym_80', 'score_asym_combined_80',
                                 'score_plus_80', 'score_plus_cond_80'])
    args = parser.parse_args()
    train_meta = train_meta.replace('new_processed', f'{args.inputs}_processed')
    val_meta = val_meta.replace('new_processed', f'{args.inputs}_processed')
    test_meta = test_meta.replace('new_processed', f'{args.inputs}_processed')
    cluster_out_path = cluster_out_path.replace('new_cluster', f'{args.inputs}_cluster')

    n_clusters = args.k
    cluster_out_path = cluster_out_path % n_clusters

    # Load meta pickle things; takes ~30s
    with open(train_meta, 'rb') as f:
        train_metas = pkl.load(f)
    
    end_points = {}
    for meta in tqdm(train_metas, 'Processing scenarios...', total=len(train_metas)):
        scenario_path = os.path.join(train_base, f'sample_{meta["scenario_id"]}.pkl')
        with open(scenario_path, 'rb') as f:
            try:
                scenario = pkl.load(f)
            except Exception as e:
                import pdb; pdb.set_trace()

        sdc_traj = scenario['track_infos']['trajs'][scenario['sdc_track_index']]
        to_predict_trajs = scenario['track_infos']['trajs'][scenario['tracks_to_predict']['track_index']]

        valid_mask = scenario['track_infos']['trajs'][:, :, -1]
        # Endpoints relative to current time step
        use_traj = (valid_mask[:, 10] == 1) & (valid_mask[:, -1] == 1)
        trajs = scenario['track_infos']['trajs'][use_traj]
        types = np.array(scenario['track_infos']['object_type'])[use_traj]
        rel_motion = trajs[:, -1, :2] - trajs[:, 10, :2]
        for obj_type, motion in zip(types, rel_motion):
            if obj_type in end_points:
                end_points[obj_type].append(motion)
            else:
                end_points[obj_type] = [motion]
        

    ret_clusters = {}
    for k, v in end_points.items():
        v = np.stack(v)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(v)
        ret_clusters[k] = kmeans.cluster_centers_
    with open(cluster_out_path, 'wb') as f:
        pkl.dump(ret_clusters, f)