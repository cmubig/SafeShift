import matplotlib.pyplot as plt
import numpy as np
import os
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_DYNAMIC'] = 'FALSE'

import pickle as pkl
import time
import pandas as pd
import torch

from natsort import natsorted
from operator import itemgetter
from tqdm import tqdm
from matplotlib import cm

from tools.scenario_identification.utils.common import (
    CACHE_DIR, VISUAL_OUT_DIR, POS_XYZ_IDX
)
from tools.scenario_identification.utils.visualization import (
    plot_static_map_infos, plot_dynamic_map_infos, get_color_map, plot_cluster_overlap,
    AGENT_COLOR
)

VISUAL_OUT_SUBDIR = os.path.join(VISUAL_OUT_DIR, f"{os.path.basename(__file__).split('.')[0]}")

def do_resample(lane, resample_level=1):
    df = pd.DataFrame(lane)
    df.index = [pd.to_datetime(1e9 * x * resample_level) for x in df.index]
    df = df.resample('1s').interpolate()
    return df.to_numpy()

# Implements closest_lanes as per the Frenet+ paper, for consistency: https://arxiv.org/pdf/2305.17965.pdf
def simple_closest_lane(trajectory, mask, lanes):
    # 1. Mask trajectory appropriately
    # 2. Calculate average distance to center line points for each traj
    # 3. Calculate "shape" similarity using "current" timestep

    lane_sims = []
    points = trajectory[mask > 0, :2]
    for ref_lane in lanes:
        # lane needs to be at least 2 points long
        if len(ref_lane) < 2:
            lane_sims.append(0)
            continue
        # Do floating point arithmetic in float64
        ref_lane = ref_lane[:, :2].astype(np.float64)
        ref_lane_segments = np.array([np.array(x) for x in zip(ref_lane[:-1], ref_lane[1:])])
        segment_lengths = np.linalg.norm(ref_lane_segments[:, 1] - ref_lane_segments[:, 0], axis=-1)
        segment_dists = np.linalg.norm(points[:, np.newaxis, np.newaxis, :] - ref_lane_segments, axis=-1).mean(axis=-1)
        closest_segment_idxs = segment_dists.argmin(axis=-1)
        closest_segments = ref_lane_segments[closest_segment_idxs]
        eps = 1e-8


        l2 = np.sum((closest_segments[:, 1]-closest_segments[:, 0])**2, axis=-1)
        l2[l2 < eps] = eps

        t = np.sum((points - closest_segments[:, 0]) * (closest_segments[:, 1] - closest_segments[:, 0]), axis=-1) / l2
        proj_points = closest_segments[:, 0] + t[:, np.newaxis] * (closest_segments[:, 1] - closest_segments[:, 0])
        on_lane_points = (t >= 0) & (t <= 1)

        # For closest_lanes, but not for cartesian2frenet, clip everything to be on lane
        lane_point_dists = np.linalg.norm(points[:, np.newaxis, :] - ref_lane, axis=-1)
        closest_point_idxs = lane_point_dists.argmin(axis=-1)
        closest_points = ref_lane[closest_point_idxs]
        proj_points[~on_lane_points] = closest_points[~on_lane_points]

        proj_dists = np.linalg.norm(points - proj_points, axis=-1)
        # Special case for when before or after the lane, only take perpendicular distance when within a segment
        # dists = segment_dists.min(axis=1)
        # dists[(on_lane_points) | (inner_proj_err)] = proj_dists[(on_lane_points) | (inner_proj_err)]
        dists = proj_dists
        dists[dists < eps] = eps

        if len(dists) == 0:
            D_1 = np.inf
            D_2 = np.inf
        elif len(dists) == 1:
            D_1 = dists[0]
            D_2 = 0
        else:
            # "Specifically, we first calculate the Euclidean distance between the 
            # vehicle’s historical trajectory and each centerline and take the reciprocal 
            # of the distance as the similarity.""
            # Eq 2
            D_1 = dists.mean()
            # Eq 3
            delta_x = points[-1] - proj_points[-1]
            # Eq 4
            inner_dists = np.linalg.norm(points - (proj_points + delta_x), axis=-1)
            inner_dists[inner_dists < eps] = eps
            D_2 = inner_dists.mean()
        total_dist = D_1 + D_2
        total_sim = 1/total_dist
        lane_sims.append(total_sim)

    assert len(lane_sims) == len(lanes), 'Mismatch in lane sim length'
    best_lane = np.argmax(lane_sims)
    return best_lane

def compute_k_closest_lanes(trajectory, mask, lanes, K = 16, resample_level = 1, threshold=10):
    """
    Inputs:
    -------
        trajectory[np.array(T, D=xyz)]: agent's trajectory.
        mask[np.array(T)]: valid data points mask
        lanes[list(np.array(Nl, D=xy))]: list of lanes in scenario.
        K[int]: K-closest lanes to keep 
    Outputs
    ------- 
        D_full[np.array(N, T, 3)]: distance matrix containing closest distance, lane index of 
            closest point, lane index in list.
        D_full[np.array(N, T, 3)]: as above but for history portion.
        
    """
    trajectory = trajectory.T[:, None, :] # (T, 1, dim) 
    T, _, _ = trajectory.shape
    N = len(lanes)

    # [distance, lane_pos_idx, proj_distance, proj_idx, proj_x, proj_y, proj_z, lane_idx]
    D_full = np.inf * np.ones(shape=(N, T, 6), dtype=np.float64) 
    # Closest distance between lane and trajectory
    for n in range(N):
        # lane = lanes[n][::resample_level]
        # hq_lane = lanes[n]
        lane = lanes[n]

        # (T, 1, dim) - (Nl, dim) --> (T, Nl, dim) --> (T, Nl)
        # dists[n] = np.linalg.norm(trajectory - lane, axis=2).min(axis=1)
        dn = np.linalg.norm(trajectory - lane, axis=2).astype(np.float64)
        dn[~mask] = np.inf

        closest_points = dn[mask].argmin(axis=1)
        closest_dists = dn[mask].min(axis=1)
        closest_pos = np.stack([lane[i] for i in closest_points]).astype(np.float64)
        if closest_dists.min() > threshold or len(trajectory[mask]) == 0:
            D_full[n, mask, 0] = closest_dists # dist value
            D_full[n, mask, 1] = closest_points # idx value
            D_full[n, mask, 2] = closest_pos[:, 0] # x val
            D_full[n, mask, 3] = closest_pos[:, 1] # y val
            D_full[n, mask, 4] = closest_pos[:, 2] # z val
            continue

        segment_points = []
        segment_ids = []
        for i, point_idx in enumerate(closest_points):
            if point_idx == 0:
                segment = (0, 1)
            elif point_idx == len(lane) - 1:
                segment = (len(lane) - 2, len(lane) - 1)
            elif dn[mask][i, point_idx - 1] < dn[mask][i, point_idx + 1]:
                segment = (point_idx - 1, point_idx)
            else:
                segment = (point_idx, point_idx + 1)
            if len(lane) == 1:
                segment = (0, 0)
            left_bound = lane[segment[0]]
            right_bound = lane[segment[1]]
            segment_points.append(np.stack([left_bound, right_bound]))
            segment_ids.append(np.array([segment[0], segment[1]]))
        segment_points = np.array(segment_points).astype(np.float64)
        segment_ids = np.array(segment_ids)

        # Following this: https://arxiv.org/pdf/2305.17965.pdf
        # and also this: https://stackoverflow.com/a/61343727/10101616
        l2 = np.sum((segment_points[:, 1]-segment_points[:, 0])**2, axis=-1)
        eps = 1e-8
        eps_mask = l2 > eps
        # if (l2 < eps).any():
        #     import pdb; pdb.set_trace()
        if eps_mask.any():
            traj_points = trajectory[mask][:, 0][eps_mask]
            # t should be between (0, 1) in order to fall within the segment, but it's okay to be outside for now
            t = np.sum((traj_points - segment_points[:, 0]) * (segment_points[:, 1] - segment_points[:, 0]), axis=-1) / l2
            proj_points = segment_points[:, 0] + t[:, np.newaxis] * (segment_points[:, 1] - segment_points[:, 0])
            along_segments = (t >= 0) & (t <= 1)

            new_dists = np.linalg.norm(proj_points - traj_points, axis=-1)
            new_idxs = t + segment_ids[:, 0][eps_mask]
            
            closest_points = closest_points.astype(np.float64)
            if along_segments.any():
                to_idx = np.arange(len(eps_mask))[eps_mask][along_segments]
                closest_dists[to_idx] = new_dists[along_segments]
                closest_points[to_idx] = new_idxs[along_segments]
                closest_pos[to_idx, 0] = proj_points[along_segments][:, 0]
                closest_pos[to_idx, 1] = proj_points[along_segments][:, 1]
                closest_pos[to_idx, 2] = proj_points[along_segments][:, 2]
        
        D_full[n, mask, 0] = closest_dists # dist value
        D_full[n, mask, 1] = closest_points # idx value
        D_full[n, mask, 2] = closest_pos[:, 0] # x val
        D_full[n, mask, 3] = closest_pos[:, 1] # y val
        D_full[n, mask, 4] = closest_pos[:, 2] # z val

    # K closest lanes --> (K, T)
    # TODO: vectorize. How to use full_k_lanes as mask?
    full_k_lanes = D_full[:, :, 0].argsort(axis=0)[:K]
    D_k = np.inf * np.ones(shape=(T, full_k_lanes.shape[0], D_full.shape[-1]))
    for t in range(T):
        k_lanes = full_k_lanes[:, t]
        D_full[k_lanes, t, -1] = k_lanes
        D_k[t] = D_full[k_lanes, t]
        
    D_k[~mask] = np.inf
    return D_k

def process_file(path: str, meta: str,  k_closest: int, plot = False, tag: str ='temp', input_id=0, resample_level=1, threshold=10):
    # Load the scenario 
    with open(path, 'rb') as f:
        scenario = pkl.load(f)
    
    # Trajectory data:
    #    center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid
    track_infos = scenario['track_infos']
    
    # Map infos:
    #   lane, road_line, road_edge, stop_sign, crosswalk, speed_bump, all_polylines
    static_map_infos = scenario['map_infos']
    # Dynamic map infos:
    #   lane_id, state, stop_point
    dynamic_map_infos = scenario['dynamic_map_infos']
        
    # min_polyline = static_map_infos['all_polylines'][:, :2].min()
    # max_polyline = static_map_infos['all_polylines'][:, :2].max()
    # min_traj = track_infos['trajs'][..., :-1][..., POS_XY_IDX].min()
    # max_traj = track_infos['trajs'][..., :-1][..., POS_XY_IDX].max()
    # max_dist = max(max_traj - min_polyline, max_polyline - min_traj)*np.sqrt(2) 

    static_map_pos = plot_static_map_infos(static_map_infos, ax=None, dim=3)
    lanes = static_map_pos['lane']
    
    # Trajectories --> (num_agents, time_steps, 9)
    trajectories = track_infos['trajs'][:, :, :-1]
    # Mask         --> (num_agents, time_steps, 1)
    valid_masks = track_infos['trajs'][:, :, -1] > 0
    
    num_agents, time_steps, dim = trajectories.shape


    out = np.zeros((num_agents, time_steps, 16, 6), dtype=np.float32) + np.inf

    #resampled_lanes = [do_resample(x, resample_level=resample_level) for x in lanes]
    resampled_lanes = lanes
    for n in range(out.shape[0]):
        mask = valid_masks[n]
        D = compute_k_closest_lanes(trajectories[n, :, POS_XYZ_IDX], mask, resampled_lanes, K=k_closest, 
                                    resample_level=resample_level, threshold=threshold).astype(np.float32)
        out[n, :, :D.shape[1], :] = D

    return out
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='~/monet_shared/shared/mtr_process')
    parser.add_argument(
        '--split', type=str, default='training', choices=['training', 'validation', 'testing'])
    parser.add_argument('--num_scenarios', type=int, default=-1)
    parser.add_argument('--k_closest', type=int, default=16)
    parser.add_argument('--resample', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=10.0)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--nproc', type=int, default=20)
    parser.add_argument('--save_freq', type=int, default=1000)
    args = parser.parse_args()

    raise NotImplementedError('Main method unfinished for closest_lanes')

    # Uses the "new" splits, from resplit.py; this way, test is labeled as well
    base = os.path.expanduser(args.base_path)
    step = 1
    if args.split == 'training':
        step = 5
        scenarios_base = f'{base}/new_processed_scenarios_training'
        scenarios_meta = f'{base}/new_processed_scenarios_training_infos.pkl'
    elif args.split == 'validation':
        scenarios_base = f'{base}/new_processed_scenarios_validation'
        scenarios_meta = f'{base}/new_processed_scenarios_val_infos.pkl'
    else:
        scenarios_base = f'{base}/new_processed_scenarios_testing'
        scenarios_meta = f'{base}/new_processed_scenarios_test_infos.pkl'

    os.makedirs(VISUAL_OUT_SUBDIR, exist_ok=True)
    CACHE_SUBDIR = os.path.join(CACHE_DIR, args.split, 'closest_lanes')
    os.makedirs(CACHE_SUBDIR, exist_ok=True)

    print(f"Loading Scenario Data...")
    with open(scenarios_meta, 'rb') as f:
        metas = pkl.load(f)[::step]
    inputs = [(f'sample_{x["scenario_id"]}.pkl', f'{scenarios_base}/sample_{x["scenario_id"]}.pkl') for x in metas]
    
    num_scenarios = len(metas)
    if args.num_scenarios != -1:
        num_scenarios = args.num_scenarios
        metas = metas[:num_scenarios]

    msg = f'Processing {args.split} split scenarios...'
    start = time.time()
    if args.parallel:
        from joblib import Parallel, delayed    
        all_outs = Parallel(n_jobs=args.nproc)(delayed(process_file)(
            path, meta, args.k_closest, args.plot, tag=f"{s.split('.')[0]}", input_id=i, resample_level=args.resample, threshold=args.threshold) 
            for i, ((s, path), meta) in tqdm(enumerate(zip(inputs, metas)), msg, total=len(metas)))
    else:
        all_outs = []
        for (s, path), meta in tqdm(zip(inputs, metas), msg, total=len(metas)):
            out = process_file(path, meta, args.k_closest, args.plot, tag=f"{s.split('.')[0]}", resample_level=args.resample, threshold=args.threshold)
            all_outs.append(out)

    # Load via: np.load(f'{CACHE_SUBDIR}/closest.npz').f.arr_0
    # with open(f'{CACHE_SUBDIR}/closest.npz', 'wb') as f:
    #     np.savez_compressed(f, all_outs)
    print(f'Saving {len(all_outs)} scenarios...')
    with open(f'{CACHE_SUBDIR}/closest_r{args.resample}.npz', 'wb') as f:
        np.savez_compressed(f, all_outs)
    
    print(f"Process took {time.time() - start} seconds.")