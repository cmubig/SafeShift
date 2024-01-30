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
    POS_XYZ_IDX, CACHE_DIR, POS_XY_IDX, VEL_XY_IDX
)

VISUAL_OUT_DIR = 'out/scenario_id/vis'
VISUAL_OUT_SUBDIR = os.path.join(VISUAL_OUT_DIR, f"{os.path.basename(__file__).split('.')[0]}")

from tools.scenario_identification.meat_lanes import build_lane_graph, plot_traj

def process_file(path: str, meta: str,  plot = False, tag: str ='temp', hist_only=False,
                 interp_vals=None):
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
    
    last_t = 91 if not hist_only else 11
    # Trajectories --> (num_agents, time_steps, 9)
    trajectories = track_infos['trajs'][:, :last_t, :-1]
    # Mask         --> (num_agents, time_steps, 1)
    valid_masks = track_infos['trajs'][:, :last_t, -1] > 0
    
    # For hist_only
    fut_trajs = track_infos['trajs'][:, last_t:, :-1]
    fut_masks = track_infos['trajs'][:, last_t:, -1] > 0
    
    num_agents, time_steps, dim = trajectories.shape
    
    outs = []
    for n in range(num_agents):
        mask = valid_masks[n]
        fut_mask = fut_masks[n]
        fut_traj = fut_trajs[n]
        interp_val = interp_vals[n]
        interp_fut = interp_val[last_t:]

        if not mask.any():
            out = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
            outs.append(out)
            continue

        ac = 0
        fig, ax = plt.subplots(1, 2, figsize=(5 * 2, 5 * 1))
        
        static_map_pos = plot_static_map_infos(static_map_infos, ax)
        
        traj = interp_val[:, :2]
        traj_given = traj[interp_val[:, -1] == 1]
        traj_interp = traj[interp_val[:, -1] == 0]
        ax[ac].scatter(traj_given[:, 0], traj_given[:, 1], s=2, color='blue', marker='.', alpha=0.25)
        ax[ac].scatter(traj_interp[:, 0], traj_interp[:, 1], s=2, color='green', marker='.', alpha=0.25)
        extent = ax[ac].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(os.path.join(VISUAL_OUT_SUBDIR, f"{tag}_{n}.png"), dpi=300, bbox_inches=extent)
        plt.show()
        plt.close()
        # Whether or not the projections were done via Frenet in lane
        out = np.array([], dtype=np.float32)
        
        outs.append(out)

    return np.array(outs)
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='~/monet_shared/shared/mtr_process')
    parser.add_argument('--cache_path', type=str, default='/av_shared_ssd/scenario_id/cache')
    parser.add_argument('--split', type=str, default='training', choices=['training', 'validation', 'testing'])
    parser.add_argument('--num_scenarios', type=int, default=-1)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--dist_threshold', type=float, default=2)
    parser.add_argument('--thresh_iters', type=int, default=9)
    parser.add_argument('--hist_only', action='store_true')
    parser.add_argument('--prob_threshold', type=float, default=0.5)
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
    if args.load_cache:
        # Train takes ~200s for full, ~60s for hist only
        # Test/val takes ~90s for full, ~30s for hist only
        print("Loading cache")
        file_name = 'interp' if not args.hist_only else 'interp_hist'
        shard_suffix = f'_shard{args.shard_idx}_{args.shards}' if args.shards > 1 else ''
        interp_vals_filepath = os.path.join(args.cache_path, f"{args.split}/frenet/{file_name}{shard_suffix}.npz")
        interp_vals = np.load(interp_vals_filepath, allow_pickle=True)['arr_0']
        print(f"Loading closest lanes took {time.time() - start} seconds")
    else:
        raise NotImplementedError('Frenet interp lanes must be cached')

    os.makedirs(VISUAL_OUT_SUBDIR, exist_ok=True)
    CACHE_SUBDIR = os.path.join(CACHE_DIR, args.split, 'frenet')
    os.makedirs(CACHE_SUBDIR, exist_ok=True)

    # Load meta pickle things; takes ~30s
    print(f"Loading Scenario Data...")
    with open(scenarios_meta, 'rb') as f:
        metas = pkl.load(f)[::step]
    inputs = [(f'sample_{x["scenario_id"]}.pkl', f'{scenarios_base}/sample_{x["scenario_id"]}.pkl') for x in metas]

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
        interp_vals = interp_vals[:num_scenarios]

    msg = f'Processing {args.split} split scenarios...'
    start = time.time()

    if args.parallel:
        from joblib import Parallel, delayed    
        all_outs = Parallel(n_jobs=args.nproc, batch_size=4)(delayed(process_file)(
            path, meta, args.plot, tag=f"{s.split('.')[0]}",
            hist_only=args.hist_only, interp_vals=interp_info)
            for (s, path), meta, interp_info in tqdm(zip(inputs, metas, interp_vals), msg, total=len(metas)))
    else:
        all_outs = []
        for (s, path), meta, interp_info in tqdm(zip(inputs, metas, interp_vals), msg, total=len(metas)):
            out = process_file(path, meta, args.plot, tag=f"{s.split('.')[0]}",
                        hist_only=args.hist_only, interp_vals=interp_info)
            all_outs.append(out)

    print(f"Process took {time.time() - start} seconds.")

