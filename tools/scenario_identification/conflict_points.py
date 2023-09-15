import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import time

from tqdm import tqdm

from utils.common import load_infos, find_conflict_points, VISUAL_OUT_DIR, CACHE_DIR
from utils.visualization import plot_dynamic_map_infos, plot_static_map_infos

def process_file(path: str, meta: str, plot: bool = False, tag: str ='temp'):
    # Load the scenario 
    with open(path, 'rb') as f:
        scenario = pkl.load(f)

    static_map_infos, dynamic_map_infos = scenario['map_infos'], scenario['dynamic_map_infos']
    conflict_points = find_conflict_points(static_map_infos, dynamic_map_infos)
     
    if plot:
        num_windows = 3
        point_size = 1
        color = 'blue'
        alpha = 0.5
        fig, ax = plt.subplots(1, num_windows, figsize=(5 * num_windows, 5 * 1))
            
        plot_static_map_infos(static_map_infos, ax)
        plot_dynamic_map_infos(dynamic_map_infos, ax)

        static_cp = conflict_points['static']
        if len(static_cp) > 0:
            ax[0].scatter(static_cp[:, 0], static_cp[:, 1], color=color, alpha=alpha, s=point_size)
        ax[0].set_title('Static')

        intersections = conflict_points['lane_intersections']
        if len(intersections) > 0:
            ax[1].scatter(intersections[:, 0], intersections[:, 1], color=color, alpha=alpha, s=point_size)
        ax[1].set_title('Lane Intersections')

        dynamic_cp = conflict_points['dynamic']
        if len(dynamic_cp) > 0:
            ax[2].scatter(dynamic_cp[:, 0], dynamic_cp[:, 1], color=color, alpha=alpha, s=point_size)
        ax[2].set_title('Dynamic')

        for a in ax.reshape(-1):
            a.set_xticks([])
            a.set_yticks([])

        plt.suptitle(f'Conflict Points')
        plt.savefig(os.path.join(VISUAL_OUT_SUBDIR, f"{tag}.png"), dpi=500, bbox_inches='tight')
        plt.show()
        plt.close()

    return conflict_points

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='~/monet_shared/shared/mtr_process')
    parser.add_argument('--split', type=str, default='train', choices=['training', 'validation', 'testing'])
    parser.add_argument('--prob_threshold', type=float, default=0.5)
    parser.add_argument('--dist_threshold', type=float, default=2)
    parser.add_argument('--num_scenarios', type=int, default=-1)
    parser.add_argument('--thresh_iters', type=int, default=9)
    parser.add_argument('--nproc', type=int, default=10)
    parser.add_argument('--shard_idx', type=int, nargs='+',required=True)
    parser.add_argument('--shards', type=int, default=10)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--load_cache', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--use_interp', action='store_true')
    args = parser.parse_args()

    VISUAL_OUT_SUBDIR = os.path.join(VISUAL_OUT_DIR, args.split, f"{__file__.split('.')[0]}")
    os.makedirs(VISUAL_OUT_SUBDIR, exist_ok=True)
    
    cache_path = os.path.join(CACHE_DIR, args.split)

    infos = load_infos(
        base_path=args.base_path, cache_path=cache_path, split=args.split, shard_idx=args.shard_idx, 
        shards=args.shards, num_scenarios=args.num_scenarios)
    metas, inputs = infos['metas'], infos['inputs']

    zipped = zip(inputs, metas)

    msg = f'Processing {args.split} split scenarios...'
    start = time.time()
    if args.parallel:
        from joblib import Parallel, delayed    
        all_outs = Parallel(n_jobs=args.nproc, batch_size=4)(delayed(process_file)(
            path, meta, plot=args.plot, tag=f"{VISUAL_OUT_SUBDIR}/{s.split('.')[0]}")
            for (s, path), meta in tqdm(zipped, desc=msg, total=len(metas)))
    else:
        all_outs = []
        for (s, path), meta in tqdm(zipped, msg, total=len(metas)):
            out = process_file(path, meta, plot=args.plot, tag=f"{VISUAL_OUT_SUBDIR}/{s.split('.')[0]}")
            all_outs.append(out)
    print(f"\tProcessing took {time.time() - start} seconds.")

    CACHE_SUBDIR = os.path.join(CACHE_DIR, args.split, f"{__file__.split('.')[0]}")
    os.makedirs(CACHE_SUBDIR, exist_ok=True)

    with open(f"{CACHE_SUBDIR}/{__file__.split('.')[0]}_shard{args.shard_idx[0]}_10.npz", 'wb') as f:
        np.savez_compressed(f, all_outs)
    print("Done.")