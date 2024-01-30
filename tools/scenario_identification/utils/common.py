import argparse
import itertools
import pdb
import os
import pickle as pkl
import pandas as pd
import pingouin as pg

from tqdm import tqdm
from matplotlib import pyplot as plt
from natsort import natsorted
from operator import itemgetter
import uuid
import time

from scipy import signal
from scipy import stats

from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt

import numpy as np
import itertools

CACHE_DIR = f"/av_shared_ssd/scenario_id/cache/"
VISUAL_OUT_DIR = f"/av_shared_ssd/scenario_id/vis/"
MEASUREMENT_OUT_DIR = f"/av_shared_ssd/scenario_id/measurements/"
STATS_OUT_DIR = f"/av_shared_ssd/scenario_id/stats/"

# center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y
AGENT_DIMS = [False, False, False, True, True, True, False, False, False]
HEADING_IDX = [False, False, False, False, False, False, True, False, False]
POS_XY_IDX = [True, True, False, False, False, False, False, False, False]
POS_XYZ_IDX = [True, True, True, False, False, False, False, False, False]
VEL_XY_IDX = [False, False, False, False, False, False, False, True, True]

# Interpolated stuff
IPOS_XY_IDX = [True, True, False, False, False, False, False]
IPOS_SDZ_IDX = [False, False, True, True, True, False, False]
IPOS_SD_IDX = [False, False, False, True, True, False, False]
ILANE_IDX = [False, False, False, False, False, True, False]
IVALID_IDX = [False, False, False, False, False, False, True]

AGENT_TYPE_MAP = {'TYPE_VEHICLE': 0, 'TYPE_PEDESTRIAN': 1, 'TYPE_CYCLIST': 2}
AGENT_NUM_TO_TYPE = {0: 'TYPE_VEHICLE', 1: 'TYPE_PEDESTRIAN', 2: 'TYPE_CYCLIST'}

def km_to_miles(km):
    return km / 1.60934

def meters_to_miles(meters):
    return meters / 1609.34

def mph_to_ms(mph):
    return 1609.34 * mph / 3600.0

def load_infos(
    base_path: str, cache_path: str, split: str, shard_idx: list, shards: int, num_scenarios: int, 
    load_type: str = 'gt', load_lane_cache: bool = False, load_conflict_points: bool = False,
    load_cluster_anomaly: bool = False
):  
    supported_load_types = ['gt', 'fe', 'ho']
    assert load_type in supported_load_types, f"Load type {load_type} not in {supported_load_types}"

    shard_suffix = f'_shard{shard_idx[0]}_{shards}' if shards > 1 else ''

    cache_path = cache_path.replace('/av_shared_ssd/scenario_id', 
                                    os.path.expanduser('~/monet_shared/shared/scenario_identification'))
    if load_type == 'gt':
        interp_trajs_filepath = os.path.join(cache_path, f"frenet/interp{shard_suffix}.npz")
        closest_lanes_filepath = os.path.join(cache_path, f"meat_lanes/lanes{shard_suffix}.npz")
        cluster_anomaly_filepath = os.path.join(cache_path, f"cluster_anomaly/anomaly{shard_suffix}.npz")
    elif load_type == 'ho':
        interp_trajs_filepath = os.path.join(cache_path, f"frenet/interp_hist{shard_suffix}.npz")
        closest_lanes_filepath = os.path.join(cache_path, f"meat_lanes/lanes_hist{shard_suffix}.npz")
        cluster_anomaly_filepath = os.path.join(cache_path, f"cluster_anomaly/anomaly_hist{shard_suffix}.npz")
    else: # fe
        interp_trajs_filepath = os.path.join(cache_path, f"frenet/interp_hist{shard_suffix}.npz")
        closest_lanes_filepath = os.path.join(cache_path, f"meat_lanes/lanes_hist{shard_suffix}.npz")
        cluster_anomaly_filepath = os.path.join(cache_path, f"cluster_anomaly/anomaly_extrap{shard_suffix}.npz")

    # Loading interpolated trajectories 
    print(f"Loading {split} infos - shard {shard_idx}")

    start = time.time()
    print(f"Loading interpolated trajectories cache from {interp_trajs_filepath}...")
    interp_trajectories = np.load(interp_trajs_filepath, allow_pickle=True)['arr_0']
    print(f"\tLoading took {time.time() - start} seconds")

    closest_lanes = None
    if load_lane_cache:
        start = time.time()
        # Train takes ~200s for full, ~60s for hist only
        # Test/val takes ~90s for full, ~30s for hist only
        print(f"Loading lane cache from {closest_lanes_filepath}..") 
        closest_lanes = np.load(closest_lanes_filepath, allow_pickle=True)['arr_0']
        # closest_lanes_metapath = os.path.join(cache_path, f"meat_lanes/{file_name}_meta{shard_suffix}.csv")
        # all_meta = pd.read_csv(closest_lanes_metapath)
        # all_meta = all_meta.drop(columns='Unnamed: 0')
        closest_lanes = [info[-1] for info in closest_lanes]
        print(f"\tLoading took {time.time() - start} seconds")

    # Loading conflict points
    conflict_points = None
    if load_conflict_points:
        start = time.time()
        conflict_points_filepath = os.path.join(cache_path, f"conflict_points/conflict_points{shard_suffix}.npz")
        print(f"Loading conflict points cache from {conflict_points_filepath}...")
        conflict_points = np.load(conflict_points_filepath, allow_pickle=True)['arr_0']
        print(f"\tLoading took {time.time() - start} seconds")

    # Loading conflict points
    cluster_anomaly = None
    if load_cluster_anomaly:
        start = time.time()
        print(f"Loading cluster anomaly cache from {cluster_anomaly_filepath}...")
        cluster_anomaly = np.load(cluster_anomaly_filepath, allow_pickle=True)['arr_0']
        print(f"\tLoading took {time.time() - start} seconds")

    base = os.path.expanduser(base_path)
    step = 1
    if split == 'training':
        step = 5
        scenarios_base = f'{base}/joint_original'
        scenarios_meta = f'{base}/new_processed_scenarios_training_infos.pkl'
    elif split == 'validation':
        scenarios_base = f'{base}/joint_original'
        scenarios_meta = f'{base}/new_processed_scenarios_val_infos.pkl'
    else:
        scenarios_base = f'{base}/joint_original'
        scenarios_meta = f'{base}/new_processed_scenarios_test_infos.pkl'

    # Load meta pickle things; takes ~30s
    print(f"Loading scenario data from {scenarios_meta}...")
    start = time.time()
    with open(scenarios_meta, 'rb') as f:
        metas = pkl.load(f)[::step]
    inputs = [(f'sample_{x["scenario_id"]}.pkl', f'{scenarios_base}/sample_{x["scenario_id"]}.pkl') for x in metas]
    print(f"\tLoading took {time.time() - start} seconds")

    if shards > 1:
        n_per_shard = np.ceil(len(metas)/shards)
        shard_start = int(n_per_shard*shard_idx[0])
        shard_end = int(n_per_shard*(shard_idx[-1] + 1))
        metas = metas[shard_start:shard_end]
        inputs = inputs[shard_start:shard_end]

    if num_scenarios != -1:
        metas = metas[:num_scenarios]
        inputs = inputs[:num_scenarios]
        if not closest_lanes is None:
            closest_lanes = closest_lanes[:num_scenarios]
            
        if not interp_trajectories is None:
            interp_trajectories = interp_trajectories[:num_scenarios]

        if not conflict_points is None:
            conflict_points = conflict_points[:num_scenarios]
        
        if not cluster_anomaly is None:
            cluster_anomaly = cluster_anomaly[:num_scenarios]
    
    # all_metas = None
    # if not closest_lanes is None:
    #     agents_per_scene = np.array([len(x) for x in closest_lanes])
    #     n_agents = np.sum(agents_per_scene)
    #     all_meta = all_meta[:n_agents]
    #     tots1 = np.cumsum(agents_per_scene)
    #     tots0 = np.array([0] + [*tots1[:-1]])
    #     all_metas = [all_meta[x0:x1] for x0, x1 in zip(tots0, tots1)]
    
    if closest_lanes is None:
        closest_lanes = [0 for _ in range(len(metas))]

    if conflict_points is None:
        conflict_points = [0 for _ in range(len(metas))]

    if cluster_anomaly is None:
        cluster_anomaly = [0 for _ in range(len(metas))]
    
    return {
        "metas": metas, 
        "inputs": inputs, 
        "closest_lanes": closest_lanes, 
        # "closest_lanes_metas": all_metas, 
        "interp_trajectories": interp_trajectories, 
        "conflict_points": conflict_points,
        "cluster_anomaly": cluster_anomaly
    }

def load_features(
    base_path: str, load_type: str, shard_idxs: list = [0], shards: int = 10, load_frenetp: bool = False, 
    load_individual_state: bool = False, load_interaction_state: bool = False,  
):
    if load_type == 'gt':
        frenet_prefix = 'frenetp_features_interp'
    else:
        frenet_prefix = 'frenetp_features_hist'

    frenetp_features = None
    if load_frenetp:
        print("Loading frenetp features...")
        start = time.time()
        frenetp_features = np.concatenate([
            np.load(f"{base_path}/{frenet_prefix}_shard{shard_idx}_{shards}.npz", allow_pickle=True)['arr_0']
            for shard_idx in range(shard_idxs[0], shard_idxs[-1]+1)])
        print(f"\tLoading took {time.time() - start} seconds")

    load_individual_state

    return {
        "frenetp_features": frenetp_features
    }


def find_conflict_points(
    static_map_info, dynamic_map_info, intersection_threshold = 0.05, resample_factor = 3):
    P = static_map_info['all_polylines']

    # Static Conflict Points: Crosswalks, Speed Bumps and Stop Signs
    static_conflict_points = []
    for cp in static_map_info['crosswalk'] + static_map_info['speed_bump']:
        start, end = cp['polyline_index']
        points = P[start:end][:, :3]
        points = signal.resample(points, points.shape[0] * resample_factor)
        static_conflict_points.append(points)

    for cp in static_map_info['stop_sign']:
        start, end = cp['polyline_index']
        points = P[start:end][:, :3]
        static_conflict_points.append(points)

    if len(static_conflict_points) > 0:
        static_conflict_points = np.concatenate(static_conflict_points)

    # Lane Intersections
    lane_infos = static_map_info['lane']
    lanes = [P[li['polyline_index'][0]:li['polyline_index'][1]][:, :3] for li in lane_infos]
    # lanes = []
    # for lane_info in static_map_info['lane']:
    #     start, end = lane_info['polyline_index']
    #     lane = P[start:end]
    #     lane = signal.resample(lane, lane.shape[0] * resample_factor)
    #     lanes.append(lane)
    num_lanes = len(lanes)
    
    lane_combinations = list(itertools.combinations(range(num_lanes), 2))
    lane_intersections = []
    for i, j in lane_combinations:
        lane_i, lane_j = lanes[i], lanes[j]
        
        D = np.linalg.norm(lane_i[:, None] - lane_j, axis=-1)
        i_idx, j_idx = np.where(D < intersection_threshold)
        
        # TODO: determine if two lanes are consecutive, but not entry/exit lanes. If this is the 
        # case there'll be an intersection that is not a conflict point. 
        start_i, end_i = i_idx[:5], i_idx[-5:]
        start_j, end_j = j_idx[:5], j_idx[-5:]
        if (np.any(start_i < 5) and np.any(end_j > lane_j.shape[0]-5)) or (
            np.any(start_j < 5) and np.any(end_i > lane_i.shape[0]-5)):
            lanes_i_ee = lane_infos[i]['entry_lanes'] + lane_infos[i]['exit_lanes']
            lanes_j_ee = lane_infos[j]['entry_lanes'] + lane_infos[j]['exit_lanes']
            if j not in lanes_i_ee and i not in lanes_j_ee:
                continue
            
        if i_idx.shape[0] > 0:
            lane_intersections.append(lane_i[i_idx])

        if j_idx.shape[0] > 0:
            lane_intersections.append(lane_j[j_idx])
    
    if len(lane_intersections) > 0:
        lane_intersections = np.concatenate(lane_intersections)
    
    # Dynamic Conflict Points: Traffic Lights
    dynamic_conflict_points = []
    if len(dynamic_map_info['stop_point']) > 0:
        dynamic_conflict_points = np.concatenate(dynamic_map_info['stop_point'][0])

    return {
        "static":  static_conflict_points, 
        "dynamic": dynamic_conflict_points,
        "lane_intersections": lane_intersections,
    }

def compute_velocities(trajectories, f = 10):
    if trajectories is None:
        return None
    num_agents, timesteps, _ = trajectories.shape
    velocities = np.zeros(shape=(num_agents, timesteps, 2))
    velocities[:, 1:] = (trajectories[:, 1:, IPOS_XY_IDX] - trajectories[:, :-1, IPOS_XY_IDX]) * f

    # Linearly extrapolate the first velocity:
    velocities[:, 0] = 2 * f * (
        trajectories[:, 1, IPOS_XY_IDX] - trajectories[:, 0, IPOS_XY_IDX]) - velocities[:, 1]
    return velocities

def get_agent_dims(trajectories, masks):
    agent_dims = np.zeros(shape=(trajectories.shape[0], 3))
    num_agents, _, _ = trajectories.shape
    for n in range(num_agents):
        agent_dims[n] = trajectories[n, masks[n]][:, AGENT_DIMS].mean(axis=0)
    return agent_dims

def is_sharing_lane(lane_i, lane_j):
    lane_i = lane_i[np.isfinite(lane_i)]
    lane_j = lane_j[np.isfinite(lane_j)]
    return np.isin(lane_i, lane_j).any()

def is_increasing(dists, time):
    """ Check if the values are increasing over time. """
    slope, _, _, _, _ = stats.linregress(time, dists)
    return slope > 0

def compute_dists_to_conflict_points(conflict_points, trajectories, load_type):
    """ Get a list of distances between all conflict points and all agents at all points. """

    num_agents, _, _ = trajectories.shape

    D = [[] for _ in range(num_agents)]
    for n in range(num_agents):
        
        # Uses interpolated and extrapolated regions 
        if np.any(np.isnan(trajectories[n, :, IPOS_XY_IDX])):
            continue
        pos = trajectories[n, :, IPOS_XY_IDX].T

        if load_type == 'gt':
            # Uses interpolated regions only
            mask = np.where(trajectories[n, :, -1] == 1)[0]
            if mask.shape[0] == 0:
                continue
            start, end = mask[0], mask[-1]+1
            pos = trajectories[n, start:end, IPOS_XY_IDX].T

        # Distance between all trajectory time steps and list of conflict points
        D[n] = np.linalg.norm(conflict_points[:, None, :] - pos, axis=-1)
    return D

def compute_leading_agent(agent_i, agent_j,  mask = None):
    # NOTE: Could we use the direction of the lane if we know that info?
    # NOTE: I think this is not straightforward to determine without more computation, but I'm trying 
    # to make the solution as simple as possible.  
    pos_i, vel_i, heading_i, len_i, agent_type_i, is_stationary_i, in_cp_i, dist_cp_i = agent_i
    pos_j, vel_j, heading_j, len_j, agent_type_j, is_stationary_j, in_cp_j, dist_cp_j = agent_j

    if not mask is None:
        pos_i = pos_i[mask]
        pos_j = pos_j[mask]
    
    # heading in degrees, guaranteed to not differ from other person's heading by more than 45 degrees
    def angle_to(pos, heading, other_pos):
        x1, y1 = pos
        x2, y2 = other_pos
        heading = np.deg2rad(heading)
        vector_to_other_point = np.array([x2 - x1, y2 - y1])
        
        angle_to_other_point = np.arctan2(vector_to_other_point[1], vector_to_other_point[0])
        angle_difference = angle_to_other_point - heading
        
        # Adjust the angle to be between -π and π
        while angle_difference > 180:
            angle_difference -= 360
        while angle_difference < -180:
            angle_difference += 360
        
        return np.rad2deg(angle_difference)

    i_to_j_angles = np.array([angle_to(pos, heading, other_pos) for pos, heading, other_pos in zip(pos_i, heading_i, pos_j)])
    # Check if pos_j is "behind" pos_i
    i_leading = np.abs(i_to_j_angles) > 90
    # 0 -> i is leading, 1 -> j is leading
    return (~i_leading).astype(int)

    # p_i = np.linalg.norm(pos_i, axis=-1)
    # inc_i = p_i[1:] >= p_i[:-1]

    # p_j = np.linalg.norm(pos_j, axis=-1)
    # inc_j = p_j[1:] >= p_j[:-1]

    # # Agent i is leading
    # leading_vehicle = np.zeros_like(p_i)

    # for t in range(1, p_i.shape[0]):
    #     # Assume j is leading if both agents' positions are increasing, but agent j's position is 
    #     # larger than i's. 
    #     if inc_i[t-1] and inc_j[t-1] and p_j[t] > p_i[t]:
    #         leading_vehicle[t] = 1 
    #     # Assume j is leading if both agents' positions are decreasing, but agent j's position is 
    #     # smaller than j's.
    #     elif not inc_i[t-1] and not inc_j[t-1] and p_j[t] < p_i[t]:
    #         leading_vehicle[t] = 1
    #     # Unsure how to handle other cases
    #     else:
    #         # Assign previous leader
    #         if t > 0:
    #             leading_vehicle[t] = leading_vehicle[t-1]
    #         # NOTE: We could also assume that the most critical agent is the follower. For instance, 
    #         # if you have a vehicle and a pedestrian, it is probably most critical if the vehicle 
    #         # is the follower than if the pedestrian is the leader. 
    # leading_vehicle[0] = leading_vehicle[1]
    # return leading_vehicle
