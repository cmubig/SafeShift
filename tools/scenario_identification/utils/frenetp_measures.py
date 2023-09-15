import numpy as np
import os
import time

from tqdm import tqdm

from utils.common import (
    load_infos, POS_XY_IDX, VEL_XY_IDX, IPOS_XY_IDX, VISUAL_OUT_DIR, CACHE_DIR, AGENT_TYPE_MAP)
from utils.visualization import plot_static_map_infos

def compute_frenetp_features(
    scenario: str, use_interp: bool, interp_trajectories: np.array, interp_velocities: np.array, 
    hist_only: bool = False, hist_len: int = 11
):
   
    features = {
        'agent_type': np.zeros(shape=(3,), dtype=np.float32),
        'centerline': np.zeros(shape=(9,), dtype=np.float32), 
        'full_trajectory': np.zeros(shape=(11,), dtype=np.float32), 
        'hist_trajectory': np.zeros(shape=(11,), dtype=np.float32), 
    }

    track_infos = scenario['track_infos']
    trajectories = track_infos['trajs'][:, :, :-1]
    object_types = track_infos['object_type']
    valid_masks = track_infos['trajs'][:, :, -1] > 0
        
    # --------------------
    # Map-related features
     
    # Map infos: lane, road_line, road_edge, stop_sign, crosswalk, speed_bump, all_polylines
    static_map_infos = scenario['map_infos']
    static_map_pos = plot_static_map_infos(static_map_infos, ax=None, dim=3)
    
    lanes = static_map_pos['lane']
    lanes_cat = np.concatenate(lanes)
    m = np.linalg.norm(lanes_cat, axis=1)
    
    #  0 - 3: largest centerline 
    features['centerline'][:3] = lanes_cat[m.argmax()] 
    #  3 - 6: smallest centerline 
    features['centerline'][3:6] = lanes_cat[m.argmin()] 

    lane_diff = np.zeros(shape=(len(lanes), 3))
    for i, lane in enumerate(lanes):
        lane_diff[i] = np.abs(lane[0] - lane[-1])
    # 6 - 9: avg abs diff between all centerlines
    features['centerline'][6:] = lane_diff.mean(axis=0) 
    
    # ---------------------------
    # Trajectory-related features 
    num_agents, _, _ = trajectories.shape
    full_speed, hist_speed = [], []
    full_traj_diffs, hist_traj_diffs = [], []

    for n in range(num_agents):
        features['agent_type'][AGENT_TYPE_MAP[object_types[n]]] += 1

        if use_interp and not hist_only:
            speed = np.linalg.norm(interp_velocities[n], axis=-1)

            # Whole trajectory; interpolated region
            full_valid = np.where(interp_trajectories[n, :, -1] == 1)[0]
            if full_valid.shape[0] == 0:
                continue
            full_trajectory = interp_trajectories[n, full_valid[0]:(full_valid[-1]+1), IPOS_XY_IDX].T
            full_traj_diffs.append(np.abs(full_trajectory[1:] - full_trajectory[:-1]))
            full_speed.append(speed[full_valid[0]:(full_valid[-1]+1)])

            # History trajectory; interpolated region
            hist_valid = np.where(interp_trajectories[n, :hist_len, -1] == 1)[0]
            if hist_valid.shape[0] > 0:
                hist_trajectory = interp_trajectories[n, hist_valid[0]:(hist_valid[-1]+1), IPOS_XY_IDX].T
                hist_traj_diffs.append(np.abs(hist_trajectory[1:] - hist_trajectory[:-1]))
                hist_speed.append(speed[hist_valid[0]:(hist_valid[-1]+1)])

        elif use_interp and hist_only:
            if np.any(np.isnan(interp_trajectories[n, :, IPOS_XY_IDX])):
                continue

            if np.any(np.isnan(interp_velocities[n])):
                breakpoint()

            speed = np.linalg.norm(interp_velocities[n], axis=-1)

            # Whole trajectory; interpolated region
            full_trajectory = interp_trajectories[n, :, IPOS_XY_IDX].T
            full_traj_diffs.append(np.abs(full_trajectory[1:] - full_trajectory[:-1]))
            full_speed.append(speed)

            # History trajectory; interpolated region
            hist_valid = np.where(interp_trajectories[n, :hist_len, -1] == 1)[0]
            if hist_valid.shape[0] > 0:
                hist_trajectory = interp_trajectories[n, hist_valid[0]:(hist_valid[-1]+1), IPOS_XY_IDX].T
                hist_traj_diffs.append(np.abs(hist_trajectory[1:] - hist_trajectory[:-1]))
                hist_speed.append(speed[hist_valid[0]:(hist_valid[-1]+1)])
        else:
            mask = valid_masks[n]
            full_trajectory = trajectories[n, mask]
            hist_trajectory = trajectories[n, :hist_len][mask[:hist_len]]

            full_speed.append(np.linalg.norm(full_trajectory[:, VEL_XY_IDX], axis=-1))
            hist_speed.append(np.linalg.norm(hist_trajectory[:, VEL_XY_IDX], axis=-1))

            full_traj_diffs.append(
                np.abs(full_trajectory[1:, POS_XY_IDX] - full_trajectory[:-1, POS_XY_IDX]))
            hist_traj_diffs.append(
                np.abs(hist_trajectory[1:, POS_XY_IDX] - hist_trajectory[:-1, POS_XY_IDX]))
    
    if len(full_traj_diffs) > 0:
        full_traj_diffs = np.concatenate(full_traj_diffs)
        # 0 - 2: avg of abs diff between all trajectories
        features['full_trajectory'][0:2] = full_traj_diffs.mean(axis=0)  
        # 2 - 4: stddev of abs diff between all trajectories
        features['full_trajectory'][2:4] = full_traj_diffs.std(axis=0)
        # 4 - 6: max of abs diff between all trajectories
        # 6 - 8: min of abs diff between all trajectories
        m_full = np.linalg.norm(full_traj_diffs, axis=0)
        features['full_trajectory'][4:6] = full_traj_diffs[m_full.argmax()] 
        features['full_trajectory'][6:8] = full_traj_diffs[m_full.argmin()] 
        # 8: avg of cosine value between traj diffs and [1, 0] --> cos_val = a * b / |a||b|
        # 9: stddev of cosine value between traj diffs and [1, 0]
        mm_full = np.linalg.norm(full_traj_diffs, axis=1)
        c_full = np.dot(full_traj_diffs[mm_full > 0.0], np.array([1, 0])) / mm_full[mm_full > 0.0] 
        features['full_trajectory'][8] = c_full.mean()
        features['full_trajectory'][9] = c_full.std()  
        # 10: average speed
        features['full_trajectory'][10] = np.concatenate(full_speed).mean()

    if len(hist_traj_diffs) > 0:
        hist_traj_diffs = np.concatenate(hist_traj_diffs)
        features['hist_trajectory'][0:2] = hist_traj_diffs.mean(axis=0)  
        features['hist_trajectory'][2:4] = hist_traj_diffs.std(axis=0) 
    
        m_hist = np.linalg.norm(hist_traj_diffs, axis=0)
        features['hist_trajectory'][4:6] = hist_traj_diffs[m_hist.argmax()] 
        features['hist_trajectory'][6:8] = hist_traj_diffs[m_hist.argmin()] 
    
        mm_hist = np.linalg.norm(hist_traj_diffs, axis=1)
        c_hist = np.dot(hist_traj_diffs[mm_hist > 0.0], np.array([1, 0])) / mm_hist[mm_hist > 0.0]
        features['hist_trajectory'][8] = c_hist.mean()
        features['hist_trajectory'][9] = c_hist.std()  
        features['hist_trajectory'][10] = np.concatenate(hist_speed).mean()

    if np.any(np.isnan(features['full_trajectory'])):
        breakpoint()
    
    if np.any(np.isnan(features['hist_trajectory'])):
        breakpoint()

    return features

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='~/monet_shared/shared/mtr_process')
    parser.add_argument('--split', type=str, default='training', choices=['training', 'validation', 'testing'])
    parser.add_argument('--prob_threshold', type=float, default=0.5)
    parser.add_argument('--dist_threshold', type=float, default=2)
    parser.add_argument('--num_scenarios', type=int, default=-1)
    parser.add_argument('--thresh_iters', type=int, default=9)
    parser.add_argument('--nproc', type=int, default=10)
    parser.add_argument('--shard_idx', type=int, default=0)
    parser.add_argument('--shards', type=int, default=10)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--hist_only', action='store_true')
    parser.add_argument('--load_cache', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--use_interp', action='store_true')
    args = parser.parse_args()

    VISUAL_OUT_SUBDIR = os.path.join(VISUAL_OUT_DIR, args.split, f"{__file__.split('.')[0]}")
    os.makedirs(VISUAL_OUT_SUBDIR, exist_ok=True)

    cache_path = os.path.join(CACHE_DIR, args.split)

    infos = load_infos(
        base_path=args.base_path, cache_path=cache_path, split=args.split, hist_only=args.hist_only, 
        shard_idx=args.shard_idx, shards=args.shards, num_scenarios=args.num_scenarios)
    metas, inputs = infos['metas'], infos['inputs']
    
    zipped = zip(inputs, metas)

    msg = f'Processing {args.split} split scenarios...'
    start = time.time()
    if args.parallel:
        from joblib import Parallel, delayed    
        all_outs = Parallel(n_jobs=args.nproc, batch_size=4)(delayed(compute_frenetp_features)(
            path, meta, plot=args.plot, tag=f"{s.split('.')[0]}", hist_only=args.hist_only)
            for (s, path), meta in tqdm(zipped, desc=msg, total=len(metas)))
    else:
        all_outs = []
        for (s, path), meta in tqdm(zipped, msg, total=len(metas)):
            out = compute_frenetp_features(
                path, meta, plot=args.plot, tag=f"{s.split('.')[0]}", hist_only=args.hist_only)
            all_outs.append(out)
    print(f"Process took {time.time() - start} seconds.")

    CACHE_SUBDIR = os.path.join(CACHE_DIR, args.split, f"{__file__.split('.')[0]}")
    os.makedirs(CACHE_SUBDIR, exist_ok=True)
    
    if not args.hist_only:
        with open(f'{CACHE_SUBDIR}/frenetp_features.npz', 'wb') as f:
            np.savez_compressed(f, all_outs)
    else:
        with open(f'{CACHE_SUBDIR}/frenetp_features_hist.npz', 'wb') as f:
            np.savez_compressed(f, all_outs)
    print("Done.")