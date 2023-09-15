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
    POS_XYZ_IDX, CACHE_DIR, VISUAL_OUT_DIR, POS_XY_IDX, VEL_XY_IDX
)

VISUAL_OUT_SUBDIR = os.path.join(VISUAL_OUT_DIR, f"{os.path.basename(__file__).split('.')[0]}")

from tools.scenario_identification.meat_lanes import build_lane_graph

def process_file(path: str, meta: str,  plot = False, tag: str ='temp', hist_only=False,
                 interp_vals=None):
    
    with open(path, 'rb') as f:
        scenario = pkl.load(f)
    track_infos = scenario['track_infos']
    objects_type = track_infos['object_type']

    outs = []
    for interp_val, object_type in zip(interp_vals, objects_type):
        if hist_only:
            interp_val = interp_val[:11]
        speeds = np.linalg.norm(interp_val[1:, :3] - interp_val[:-1, :3], axis=-1)
        avg_speed = speeds.mean()
        min_speed = speeds.min()
        if interp_val[:, -1].sum() < 1:
            continue
        interp_region = [(vals := np.where(interp_val[:, -1] == 1))[0][0], vals[0][-1]]
        n_valid = interp_val[:, -1].sum() 
        n_interp = (interp_region[1] - interp_region[0] + 1) - n_valid.sum()
        assert n_interp >= 0, 'Somehow valid outside of interp region'
        n_extrap = len(interp_val) - n_valid - n_interp
        valid_and_lane = ((interp_val[:, -1] > 0) & (interp_val[:, -2] != np.inf)).sum()
        outs.append(np.array([interp_val[:, -1].sum(), (interp_val[:, -2] != np.inf).sum(), len(interp_val), \
                              n_interp, n_extrap, valid_and_lane, avg_speed, min_speed, object_type]))
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
    all_scores = np.concatenate(all_outs, axis=0)

    df = pd.DataFrame(all_scores, columns=['n_valid', 'n_lane', 'tot', 'n_interp', 'n_extrap', 'n_valid_and_lane',
                                           'avg_speed', 'min_speed', 'obj_type'])
    float_keys = ['n_valid', 'n_lane', 'tot', 'n_valid_and_lane', 'avg_speed', 'min_speed', 'n_interp', 'n_extrap']
    for key in float_keys:
        df[key] = df[key].astype(np.float64)

    df['stationary'] = df['avg_speed'] < 0.25
    df['lane_percent'] = df['n_lane'] / df['tot']
    df['valid_percent'] = df['n_valid'] / df['tot']
    df['lane_over_valid'] = df['n_valid_and_lane'] / df['n_valid']
    df['interp_percent'] = df['n_interp'] / df['tot']
    df['extrap_percent'] = df['n_extrap'] / df['tot']
    #df = pd.DataFrame(all_scores, columns=['n_valid', 'n_lane', 'tot'])
    print('Counts')
    print(df.groupby(['obj_type', 'stationary']).count()['n_valid'])
    print('Percent of total time steps')
    print(df.groupby(['obj_type', 'stationary']).mean()[['lane_percent', 'valid_percent', 'lane_over_valid', 'interp_percent', 'extrap_percent']])
    breakpoint()

