import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import time
import pandas as pd

from natsort import natsorted
from operator import itemgetter
from tqdm import tqdm
from matplotlib import cm

from tools.scenario_identification.utils.common import (
    plot_static_map_infos, plot_dynamic_map_infos, plot_lanes_by_distance,
    POS_XYZ_IDX, CACHE_DIR, VISUAL_OUT_DIR, POS_XY_IDX, VEL_XY_IDX, HEADING_IDX
)

VISUAL_OUT_SUBDIR = os.path.join(VISUAL_OUT_DIR, f"{os.path.basename(__file__).split('.')[0]}")

from tools.scenario_identification.closest_lanes import compute_k_closest_lanes
from tools.scenario_identification.meat_lanes import build_lane_graph, build_lane_sequences
from tools.scenario_identification.frenet_interp import process_traj

def process_file(path: str, meta: str,  plot = False, tag: str ='temp', hist_only=False):
    # Load the scenario 
    with open(path, 'rb') as f:
        scenario = pkl.load(f)
    
    # Trajectory data:
    #    center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid
    track_infos = scenario['track_infos']
    objects_type = track_infos['object_type']

    # Map infos:
    #   lane, road_line, road_edge, stop_sign, crosswalk, speed_bump, all_polylines
    static_map_infos = scenario['map_infos']
    dynamic_map_infos = scenario['dynamic_map_infos']

    static_map_pos = plot_static_map_infos(static_map_infos, ax=None, dim=3)
    lane_pos = static_map_pos['lane']
    
    lanes = static_map_infos['lane']
    lane_graph = build_lane_graph(lanes)
    lane_segments = [np.stack([lane[:-1, :], lane[1:, :]], axis=1) for lane in lane_pos]
    
    last_t = 91 if not hist_only else 11
    # Trajectories --> (num_agents, time_steps, 9)
    trajectories = track_infos['trajs'][:, :last_t, :-1]
    # Mask         --> (num_agents, time_steps, 1)
    valid_masks = track_infos['trajs'][:, :last_t, -1] > 0

    num_agents, time_steps, dim = trajectories.shape
    
    closest_lanes_idx = 0
    mask_rates = [0.3*(i+1) for i in range(3)]
    outs_per_mask = []
    for mask_rate in mask_rates:
        outs = []
        for n in range(num_agents):
            orig_mask = valid_masks[n].copy()
            new_mask = valid_masks[n].copy()
            # new_mask[new_mask] = [False if np.random.rand() < mask_rate else True for _ in range(sum(new_mask))]
            tots = sum(new_mask)
            new_mask[new_mask] = [False if i >= tots*(1 - mask_rate) else True for i in range(tots)]
            dist_thresh = 5 
            prob_thresh = 0.5
            angle_thresh = 45
            if np.any(new_mask) and np.any(orig_mask):
                trajectory_closest_lanes = compute_k_closest_lanes(trajectories[n, :, POS_XYZ_IDX], new_mask, lane_pos)
                trajectory_closest_lanes = trajectory_closest_lanes[:, :, [0, -1]]
                lane_sequences, status, n_expanded, tot_time, tree, n_valid, n_tot = \
                    build_lane_sequences(trajectory_closest_lanes[new_mask], lane_graph, lane_segments,
                                                    dist_thresh, prob_thresh, angle_thresh,
                                                    trajectories[n, :, POS_XYZ_IDX].T[new_mask],
                                                    trajectories[n, :, VEL_XY_IDX].T[new_mask],
                                                    trajectories[n, :, HEADING_IDX].T[new_mask],
                                                    track_infos, n, timeout=5)
                lane_info = lane_sequences[0] if len(lane_sequences) else None
            else:
                lane_info = None
            
            # for hist_only assessment with mask, no need to project into future
            out_linear = process_traj(trajectories[n][:, POS_XYZ_IDX], trajectories[n][:, VEL_XY_IDX], new_mask,
                                None, lane_pos, lanes, None, linear_only=True,
                                frenet_t_lower=0, frenet_t_upper=1, frenet_d_max=dist_thresh/2)
            out_frenet = process_traj(trajectories[n][:, POS_XYZ_IDX], trajectories[n][:, VEL_XY_IDX], new_mask,
                                lane_info, lane_pos, lanes, None, linear_only=False,
                                frenet_t_lower=0, frenet_t_upper=1, frenet_d_max=dist_thresh/2)
            
            diff_mask = new_mask != orig_mask
            if not sum(diff_mask) or not sum(new_mask):
                outs.append([np.inf, np.inf, mask_rate, len(diff_mask), np.inf, np.inf, np.inf])
                continue
            if np.isnan(out_linear[:, :3]).any():
                breakpoint()

            # in_lane = out_frenet[-1, -2] != np.inf
            in_lane = (out_frenet[:, -2] != np.inf).all()
            avg_speed_f = np.linalg.norm(out_frenet[1:, :3] - out_frenet[:-1, :3], axis=-1).mean()*10
            avg_speed_l = np.linalg.norm(out_linear[1:, :3] - out_linear[:-1, :3], axis=-1).mean()*10

            orig = trajectories[n][diff_mask, :3]
            linear = out_linear[diff_mask, :3]
            linear_errs = np.linalg.norm(linear - orig, axis=-1).mean()
            frenet = out_frenet[diff_mask, :3]
            frenet_errs = np.linalg.norm(frenet - orig, axis=-1).mean()
            outs.append(np.array([linear_errs, frenet_errs, mask_rate, sum(diff_mask), in_lane, avg_speed_l, avg_speed_f]))

        outs = np.array(outs)
        outs_per_mask.append(outs)

    return np.array(outs_per_mask)
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='~/monet_shared/shared/mtr_process')
    parser.add_argument('--cache_path', type=str, default='/av_shared_ssd/scenario_id/cache')
    parser.add_argument('--split', type=str, default='training', choices=['training', 'validation', 'testing'])
    parser.add_argument('--num_scenarios', type=int, default=-1)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--hist_only', action='store_true')
    parser.add_argument('--load_cache', action='store_true')
    parser.add_argument('--nproc', type=int, default=10)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--shard_idx', type=int, default=0)
    parser.add_argument('--shards', type=int, default=10)
    args = parser.parse_args()

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

    start = time.time()
    # if args.load_cache:
    #     # Train takes ~200s for full, ~60s for hist only
    #     # Test/val takes ~90s for full, ~30s for hist only
    #     print("Loading cache")
    #     file_name = 'lanes' if not args.hist_only else 'lanes_hist'
    #     shard_suffix = f'_shard{args.shard_idx}_{args.shards}' if args.shards > 1 else ''
    #     closest_lanes_filepath = os.path.join(args.cache_path, f"{args.split}/meat_lanes/{file_name}{shard_suffix}.npz")
    #     closest_lanes = np.load(closest_lanes_filepath, allow_pickle=True)['arr_0']
    #     closest_lanes_metapath = os.path.join(args.cache_path, f"{args.split}/meat_lanes/{file_name}_meta{shard_suffix}.csv")
    #     all_meta = pd.read_csv(closest_lanes_metapath)
    #     all_meta = all_meta.drop(columns='Unnamed: 0')
    #     print(f"Loading closest lanes took {time.time() - start} seconds")
    # else:
    #     raise NotImplementedError('MEAT lanes must be cached')

    os.makedirs(VISUAL_OUT_SUBDIR, exist_ok=True)
    CACHE_SUBDIR = os.path.join(CACHE_DIR, args.split, 'frenet')
    os.makedirs(CACHE_SUBDIR, exist_ok=True)

    # Load meta pickle things; takes ~30s
    print(f"Loading Scenario Data...")
    with open(scenarios_meta, 'rb') as f:
        metas = pkl.load(f)[::step]
    inputs = [(f'sample_{x["scenario_id"]}.pkl', f'{scenarios_base}/sample_{x["scenario_id"]}.pkl') for x in metas]

    # types = closest_lanes[0][0].keys()
    # valid_seqs = {k: sum([out[0][k] for out in closest_lanes]) for k in types}
    # invalid_seqs = {k: sum([out[1][k] for out in closest_lanes]) for k in types}
    # print(valid_seqs)
    # print(invalid_seqs)
    # all_meta = pd.DataFrame([x for out in closest_lanes for x in out[0]])
    # closest_lanes = [info[-1] for info in closest_lanes]
    
    if args.shards > 1:
        n_per_shard = np.ceil(len(metas)/args.shards)
        shard_start = int(n_per_shard*args.shard_idx)
        shard_end = int(n_per_shard*(args.shard_idx + 1))
        metas = metas[shard_start:shard_end]
        inputs = inputs[shard_start:shard_end]
    else:
        raise NotImplementedError('Implementation of joining shards together incomplete')
    
    num_scenarios = len(metas)
    if args.num_scenarios != -1:
        num_scenarios = args.num_scenarios
        metas = metas[:num_scenarios]
        inputs = inputs[:num_scenarios]
        #closest_lanes = closest_lanes[:num_scenarios]

    #agents_per_scene = np.array([len(x) for x in closest_lanes])
    #n_agents = np.sum(agents_per_scene)
    #all_meta = all_meta[:n_agents]
    #tots1 = np.cumsum(agents_per_scene)
    #tots0 = np.array([0] + [*tots1[:-1]])
    #all_metas = [all_meta[x0:x1] for x0, x1 in zip(tots0, tots1)]

    msg = f'Processing {args.split} split scenarios...'
    start = time.time()

    if args.parallel:
        from joblib import Parallel, delayed    
        all_outs = Parallel(n_jobs=args.nproc, batch_size=4)(delayed(process_file)(
            path, meta, args.plot, tag=f"{s.split('.')[0]}",
            hist_only=args.hist_only)
            for (s, path), meta in tqdm(zip(inputs, metas), msg, total=len(metas)))
    else:
        all_outs = []
        for (s, path), meta in tqdm(zip(inputs, metas), msg, total=len(metas)):
            out = process_file(path, meta, args.plot, tag=f"{s.split('.')[0]}",
                        hist_only=args.hist_only)
            all_outs.append(out)

    print(f"Process took {time.time() - start} seconds.")
    # N x (linear_err, frenet_err, mask_rate, n_masked)
    all_scores = np.concatenate([out[out[:, :, 0] != np.inf].reshape(-1, 7) for out in all_outs], axis=0)
    df = pd.DataFrame(all_scores, columns=['linear_err', 'frenet_err', 'mask_rate', 'n_masked', 'in_lane', 'avg_speed_linear', 'avg_speed_frenet'])
    df.in_lane = df.in_lane.astype(bool)
    print('In lane only')
    print(df[(df.in_lane)].groupby('mask_rate').mean()[['linear_err', 'frenet_err']])
    print('In lane + avg speed > 0.25')
    print(df[(df.avg_speed_linear > 0.25) & (df.in_lane)].groupby('mask_rate').mean()[['linear_err', 'frenet_err']])
    print('Overall')
    print(df.groupby('mask_rate').mean()[['linear_err', 'frenet_err']])
    breakpoint()

    # print(f'Saving {len(all_outs)} scenarios...')
    # shard_suffix = f'_shard{args.shard_idx}_{args.shards}' if args.shards > 1 else ''
    # if not args.hist_only:
    #     df.to_csv(f'{CACHE_SUBDIR}/interp_mask{shard_suffix}.csv')
    # else:
    #     df.to_csv(f'{CACHE_SUBDIR}/interp_mask_hist{shard_suffix}.csv')

