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
from skspatial.objects import Vector
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


base = os.path.expanduser('~/monet_shared/shared/mtr_process')

# Uses the "new" splits, from resplit.py; this way, test is labeled as well
train_base = f'{base}/frenet_789_processed_scenarios_training'
train_meta = f'{base}/frenet_789_processed_scenarios_training_infos.pkl'
val_base = f'{base}/frenet_789_processed_scenarios_validation'
val_meta = f'{base}/frenet_789_processed_scenarios_val_infos.pkl'
test_base = f'{base}/frenet_789_processed_scenarios_testing'
test_meta = f'{base}/frenet_789_processed_scenarios_test_infos.pkl'

joint_out = f'{base}/joint_frenet'

all_bases = ['frenet_789', 'frenet_013', 'new']

# scenario_sets = {}
# for base, metas in zip([train_base, val_base, test_base], [train_meta, val_meta, test_meta]):
#     inner_bases = [base.replace('frenet_789', b) for b in all_bases]
#     inner_metas = [metas.replace('frenet_789', b) for b in all_bases]
#     for inner_base, inner_meta in zip(inner_bases, inner_metas):
#         print('Loading meta', inner_meta)
#         tmp = np.load(inner_meta, allow_pickle=True)
#         if 'new_processed_scenarios_training_infos.pkl' in inner_meta:
#             tmp = tmp[::5]
#         scenarios = [x['scenario_id'] for x in tmp]
#         all_scenarios = set(scenarios)
#         scenario_sets[inner_base] = all_scenarios

def project_track_info(track_infos, lanes):
    # - track_infos:
    #    - trajs: (xyz, lwh [copy], heading, vel_x, vel_y, valid [copy])

    # how to find heading? Perhaps velocity vector relative to lane segment direction, as in get_angle in meat_lanes
    lane_features = []
    for lane in lanes:
        # valid_mask = traj[:, -1] > 0
        # valid_traj = traj[valid_mask]
        traj_features = []
        for traj in track_infos['trajs']:
            sd, inner_proj_err, closest_segments = simple_cartesian2frenet(traj, lane)
            sd_vel = np.zeros_like(sd)
            sd_vel[1:] = sd[1:] - sd[: -1]
            sd_vel[0] = sd_vel[1]
            real_vels = traj[:, 7:9]
            headings = []
            for real_vel, segment in zip(real_vels, closest_segments):
                current_heading_vec = Vector(real_vel)
                headings.append(current_heading_vec.angle_signed(segment[1] - segment[0]))
            headings = np.array(headings)

            # sd, sd_vel, lane_headings
            features = np.concatenate([sd, sd_vel, headings[:, np.newaxis]], axis=-1)
            traj_features.append(features)
        lane_features.append(traj_features)
    # n_center x n_traj x 91 x 5
    frenet_features = np.array(lane_features).astype(np.float32)
    frenet_out = {
        'trajs': frenet_features,
    }
    return frenet_out

def project_static(static_info, lanes):
    # - map_infos:
    #    - stop_sign: position (xyz)
    #    - all_polylines: (xyz, "xyz_dir" [see get_polyline_dir, set to zeros when length of polyline is 1], ID [copy])
    def get_polyline_dir(polyline):
        polyline_pre = np.roll(polyline, shift=1, axis=0)
        polyline_pre[0] = polyline[0]
        diff = polyline - polyline_pre
        polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
        return polyline_dir

    lane_polylines = []
    lane_stopsigns = []
    for lane in lanes:
        # Output: sd, sd_dir
        frenet_polylines = np.zeros((static_info['all_polylines'].shape[0], 4))
        for key in static_info.keys():
            if key == 'all_polylines':
                continue

            val_list = static_info[key]
            stop_sign_list = []
            for val in val_list:
                polyline_idx = val['polyline_index']
                polyline = static_info['all_polylines'][polyline_idx[0]:polyline_idx[1]]
                sd, _, _ = simple_cartesian2frenet(polyline, lane)
                sd_dir = get_polyline_dir(sd)
                frenet_polylines[polyline_idx[0]:polyline_idx[1]] = np.concatenate([sd, sd_dir], axis=-1)
                if key == 'stop_sign':
                    # adjust position
                    stop_sign_list.append(sd.squeeze())
            if key == 'stop_sign':
                stop_sign_pos = np.array(stop_sign_list)
                lane_stopsigns.append(stop_sign_pos)
        lane_polylines.append(frenet_polylines)
    all_polylines = np.array(lane_polylines).astype(np.float32)
    stopsigns = np.array(lane_stopsigns).astype(np.float32)
    frenet_out = {
        'all_polylines': all_polylines,
        'stop_sign_pos': stopsigns,
    }
    return frenet_out

def project_dynamic(dynamic_info, lanes):
    # - dynamic_map_infos:
    #    - stop_point: (xyz)
    lane_stoppoints = []
    for lane in lanes:
        stoppoints = []
        for entry in dynamic_info['stop_point']:
            out = []
            for traffic_lights in entry:
                sd, _, _ = simple_cartesian2frenet(traffic_lights, lane)
                out.append(sd)
            out = np.array(out).astype(np.float32)
            stoppoints.append(out)
        lane_stoppoints.append(stoppoints)

    frenet_out = {
        'stop_points': lane_stoppoints
    }
    return frenet_out

def process(meta, base, scenario_idx):
    scenario_path = f'{base}/sample_{meta["scenario_id"]}.pkl'
    with open(scenario_path, 'rb') as f:
        try:
            scenario = pkl.load(f)
        except Exception as e:
            import pdb; pdb.set_trace()

    # shard_info = cur_shard[shard_cnt]

    static_map_pos = plot_static_map_infos(scenario['map_infos'], ax=None, dim=3)
    lanes = static_map_pos['lane']

    trajs = scenario['track_infos']['trajs']
    to_center = scenario['tracks_to_predict']['track_index']
    to_center_trajs = trajs[to_center]

    center_lane_idxs = []
    center_lanes = []
    for traj in to_center_trajs:
        # Decide lane based only on history; fairness for val/test
        hist_traj = traj[:11]
        closest_lane = simple_closest_lane(hist_traj[:, :2], hist_traj[:, -1], lanes)
        ref_lane = lanes[closest_lane]
        center_lane_idxs.append(closest_lane)
        center_lanes.append(ref_lane)
    
    # Fields which need projection:
    # - track_infos, map_infos, dynamic_map_infos
    proj_tracks = project_track_info(scenario['track_infos'], center_lanes)
    proj_static = project_static(scenario['map_infos'], center_lanes)
    proj_dynamic = project_dynamic(scenario['dynamic_map_infos'], center_lanes)
    combined = {
        'lanes': center_lanes,
        'lane_idxs': center_lane_idxs,
    }
    combined.update(proj_tracks)
    combined.update(proj_static)
    combined.update(proj_dynamic)

    out_path = f'{joint_out}/frenet_{scenario["scenario_id"]}.pkl'
    with open(out_path, 'wb') as f:
        pkl.dump(combined, f)
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=64, help='how many clusters?')
    parser.add_argument('--inputs', type=str, default='frenet_013', choices=all_bases)
    parser.add_argument('--num_scenarios', type=int, default=-1)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--nproc', type=int, default=20)
    args = parser.parse_args()
    n_clusters = args.k

    out_tag = args.inputs
    train_base = train_base.replace('frenet_789', out_tag)
    train_meta = train_meta.replace('frenet_789', out_tag)
    val_base = val_base.replace('frenet_789', out_tag)
    val_meta = val_meta.replace('frenet_789', out_tag)
    test_base = test_base.replace('frenet_789', out_tag)
    test_meta = test_meta.replace('frenet_789', out_tag)

    os.makedirs(joint_out, exist_ok=True)

    # Load meta pickle things; takes ~30s
    for meta_path, base in zip([train_meta, val_meta, test_meta], [train_base, val_base, test_base]):
        with open(meta_path, 'rb') as f:
            metas = pkl.load(f)
        # only use 20% of real training data
        if base == train_base and out_tag == 'new':
            metas = metas[::5]

        if args.num_scenarios != -1:
            metas = metas[:args.num_scenarios]

        if args.parallel:
            from joblib import Parallel, delayed    
            with tqdm_joblib(tqdm(desc='Assigning singles', leave=False, total=len(metas), disable=False)) as progress_bar:
                all_out = Parallel(n_jobs=args.nproc)(delayed(process)(meta, base, scenario_idx)
                    for scenario_idx, meta in enumerate(metas))
        else:
            all_out = []
            for scenario_idx, meta in enumerate(tqdm(metas, 'Processing scenarios...', total=len(metas))):
                out = process(meta, base, scenario_idx)
                all_out.append(out)
        
    print('Done')