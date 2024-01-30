import argparse
import pdb
import os
import pickle as pkl
import pandas as pd
import pingouin as pg
import glob

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


base = os.path.expanduser('/av_shared_ssd/mtr_process_ssd')

# Uses the "new" splits, from resplit.py; this way, test is labeled as well
train_base = f'{base}/joint_original'
train_meta = f'{base}/frenet_789_processed_scenarios_training_infos.pkl'

cluster_out_path = '../../data/waymo/frenet_789_xy_cluster_%i_center_dict.pkl'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=64, help='how many clusters?')
    parser.add_argument('--inputs', type=str, default='frenet_789', choices=['new', 'frenet_013', 'frenet_789'])
    parser.add_argument('--num_scenarios', type=int, default=-1)
    args = parser.parse_args()
    n_clusters = args.k
    cluster_out_path = cluster_out_path % n_clusters

    out_tag = args.inputs
    train_base = train_base.replace('frenet_789', out_tag)
    train_meta = train_meta.replace('frenet_789', out_tag)
    cluster_out_path = cluster_out_path.replace('frenet_789', out_tag)

    # Load meta pickle things; takes ~30s
    with open(train_meta, 'rb') as f:
        train_metas = pkl.load(f)
    if out_tag == 'new':
        train_metas = train_metas[::5]

    if args.num_scenarios != -1:
        train_metas = train_metas[:args.num_scenarios]
    
    end_points = {}
    for meta in tqdm(train_metas, 'Processing scenarios...', total=len(train_metas)):
        scenario_path = f'{train_base}/sample_{meta["scenario_id"]}.pkl'
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