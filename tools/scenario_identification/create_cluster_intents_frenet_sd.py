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
from collections import deque
from tools.scenario_identification.utils.systems import tqdm_joblib
from tools.scenario_identification.utils.visualization import plot_static_map_infos
from frenet_interp import cartesian2frenet, frenet2cartesian, simple_cartesian2frenet, simple_frenet2cartesian
from closest_lanes import simple_closest_lane


base = os.path.expanduser('/av_shared_ssd/mtr_process_ssd')

# Uses the "new" splits, from resplit.py; this way, test is labeled as well
train_base = f'{base}/joint_original'
train_meta = f'{base}/frenet_789_processed_scenarios_training_infos.pkl'

# Performs identical process to the create_cluster_intents_frenet script, but in Frenet space
cluster_out_path = '../../data/waymo/frenet_789_sd_cluster_%i_center_dict.pkl'

def process(meta, scenario_idx):
    scenario_path = f'{train_base}/sample_{meta["scenario_id"]}.pkl'
    with open(scenario_path, 'rb') as f:
        try:
            scenario = pkl.load(f)
        except Exception as e:
            import pdb; pdb.set_trace()

    # shard_info = cur_shard[shard_cnt]

    sdc_traj = scenario['track_infos']['trajs'][scenario['sdc_track_index']]
    to_predict_trajs = scenario['track_infos']['trajs'][scenario['tracks_to_predict']['track_index']]

    valid_mask = scenario['track_infos']['trajs'][:, :, -1]
    # Endpoints relative to current time step
    # shard_mask = np.isfinite(shard_info[:, :, -2])
    # use_mask = (valid_mask > 0) & shard_mask
    use_mask = (valid_mask > 0)
    use_traj = (use_mask[:, 10] == 1) & (use_mask[:, -1] == 1)

    static_map_pos = plot_static_map_infos(scenario['map_infos'], ax=None, dim=3)
    lanes = static_map_pos['lane']

    trajs = scenario['track_infos']['trajs'][use_traj]
    types = np.array(scenario['track_infos']['object_type'])[use_traj]
    rel_motions = []
    for traj_idx, traj in enumerate(trajs):
        traj_mask = traj[:, -1] > 0
        valid_traj = traj[traj_mask]
        closest_lane = simple_closest_lane(valid_traj[:, :2], valid_traj[:, -1], lanes)
        ref_lane = lanes[closest_lane]
        plt.clf()
        sd, inner_proj_err, closest_segments = simple_cartesian2frenet(valid_traj, ref_lane)
        # TODO: explore angle bisection instead of just closest
        # xy = simple_frenet2cartesian(sd, ref_lane)
        # recon_err = np.linalg.norm(xy - valid_traj[:, :2], axis=-1).mean() 
        # if recon_err > 1e-1:
        #     plt.legend()
        #     plt.savefig('tmp_lane.png')
        #     breakpoint()
        # continue
        sd_out = np.ones_like(traj[:, :2]) * np.nan
        sd_out[traj_mask] = sd
        rel_motion = sd_out[-1] - sd_out[10]
        assert not (np.isnan(rel_motion)).any(), 'Nan made its way in somehow'
        rel_motions.append(rel_motion)

    rel_motions = np.array(rel_motions)
    assert len(types) == len(rel_motions), 'Mismatch'
    return rel_motions, types

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=64, help='how many clusters?')
    parser.add_argument('--inputs', type=str, default='frenet_789', choices=['new', 'frenet_013', 'frenet_789'])
    parser.add_argument('--num_scenarios', type=int, default=-1)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--nproc', type=int, default=20)
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

    if args.parallel:
        from joblib import Parallel, delayed    
        with tqdm_joblib(tqdm(desc='Assigning singles', leave=False, total=len(train_metas), disable=False)) as progress_bar:
            all_out = Parallel(n_jobs=args.nproc)(delayed(process)(meta, scenario_idx)
                for scenario_idx, meta in enumerate(train_metas))
    else:
        all_out = []
        for scenario_idx, meta in enumerate(tqdm(train_metas, 'Processing scenarios...', total=len(train_metas))):
            rel_motions, types = process(meta, scenario_idx)
            all_out.append((rel_motions, types))
    

    end_points = {}
    for rel_motions, types in all_out:
        for obj_type, motion in zip(types, rel_motions):
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