import argparse
import pdb
import datetime
import os
import pickle as pkl
import logging
from pathlib import Path
import numpy as np
import torch
import hashlib
import sys
import json
import io
import time
import contextlib

from tqdm import tqdm
from copy import deepcopy
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import silhouette_score
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import dtw
from natsort import natsorted
import mdtraj as md
from scipy.interpolate import interp1d
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform
from kneed import KneeLocator
import pandas as pd
import mdtraj.geometry.alignment as align
from mdtraj.utils import ensure_type
import pyhsmm

# Force using CPU instead of GPU, since calling waymo_eval so often...
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from mtr.config import cfg, cfg_from_yaml_file
from mtr.utils import common_utils

from tools.scenario_identification.utils.common import (
    POS_XYZ_IDX, CACHE_DIR, VISUAL_OUT_DIR, POS_XY_IDX, VEL_XY_IDX, HEADING_IDX 
)
from primitives import get_encounters, get_singles, get_velocities

VISUAL_OUT_SUBDIR = os.path.join(VISUAL_OUT_DIR, f"{os.path.basename(__file__).split('.')[0]}")

def load_base_with_prims_clusters(split, base_path, cache_path, shards, num_scenarios, hist_only, extrap, shard_idx=0):
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

    start = time.time()
    print(f"Loading {split} Scenario Data...")
    with open(scenarios_meta, 'rb') as f:
        metas = pkl.load(f)[::step]
    inputs = [(f'sample_{x["scenario_id"]}.pkl', f'{scenarios_base}/sample_{x["scenario_id"]}.pkl') for x in metas]
    print(f"Process took {time.time() - start} seconds.")

    start = time.time()
    print(f"Loading cache {shard_idx} of {10}")
    file_name = 'interp' if (not hist_only and not extrap) else 'interp_hist'
    shard_suffix = f'_shard{shard_idx}_{shards}' if shards > 1 else ''
    interp_vals_filepath = os.path.join(cache_path, f"{split}/frenet/{file_name}{shard_suffix}.npz")
    interp_vals = np.load(interp_vals_filepath, allow_pickle=True)['arr_0']
    print(f"Loading shards took {time.time() - start} seconds")

    old_cache_path = os.path.expanduser('~/monet_shared/shared/scenario_identification/cache')
    start = time.time()
    print(f"Loading primitive cache {shard_idx} of {10}")
    file_name = 'prims_hist' if hist_only else 'prims_extrap' if extrap else 'prims'
    shard_suffix = f'_shard{shard_idx}_{shards}' if shards > 1 else ''
    primitives_filepath = os.path.join(old_cache_path, f"{split}/primitives_spline/{file_name}{shard_suffix}.npz")
    primitives = np.load(primitives_filepath, allow_pickle=True)['arr_0']
    print(f"Loading primitive shards took {time.time() - start} seconds")

    start = time.time()
    print(f"Loading single cluster cache {shard_idx} of {10}")
    file_name = 'single_center_dists_hist' if hist_only else 'single_center_dists_extrap' if extrap else 'single_center_dists'
    shard_suffix = f'_shard{shard_idx}_{shards}' if shards > 1 else ''
    clusters_filepath = os.path.join(old_cache_path, f"{split}/cluster/{file_name}{shard_suffix}.npz")
    clusters_single = np.load(clusters_filepath, allow_pickle=True)['arr_0']
    print(f"Loading single cluster shards took {time.time() - start} seconds")

    start = time.time()
    print(f"Loading pair cluster cache {shard_idx} of {10}")
    file_name = 'pair_center_dists_hist' if hist_only else 'pair_center_dists_extrap' if extrap else 'pair_center_dists'
    shard_suffix = f'_shard{shard_idx}_{shards}' if shards > 1 else ''
    clusters_filepath = os.path.join(old_cache_path, f"{split}/cluster/{file_name}{shard_suffix}.npz")
    clusters_pair = np.load(clusters_filepath, allow_pickle=True)['arr_0']
    print(f"Loading pair cluster shards took {time.time() - start} seconds")

    n_per_shard = np.ceil(len(metas)/10)
    shard_start = int(n_per_shard*shard_idx)
    shard_end = int(n_per_shard*(shard_idx + 1))
    metas = metas[shard_start:shard_end]
    inputs = inputs[shard_start:shard_end]

    tot_scenarios = len(metas)
    if num_scenarios != -1:
        tot_scenarios = num_scenarios
        metas = metas[:tot_scenarios]
        inputs = inputs[:tot_scenarios]
        interp_vals = interp_vals[:tot_scenarios]

    return metas, inputs, interp_vals, primitives, clusters_single, clusters_pair

def calc_anomaly(args, base_path, cache_path, num_scenarios=-1, shards=10, num_shards=-1,
                  hist_only=False, extrap=False, min_timesteps=5, load_dists=False):


    def process(metas, inputs, interp_vals, primitives, clusters_single, clusters_pair):
        single_outs = []
        pair_outs = []
        pair_id_outs = []
        for agent_idx, agent_traj in enumerate(interp_vals):
            prim_singles, prim_pairs = primitives['singles'], primitives['pairs']
            if not len(prim_singles):
                prim_singles = np.empty((0, 4))
            if not len(prim_pairs):
                prim_pairs = np.empty((0, 5))

            single_len_mask = prim_singles[:, 2] - prim_singles[:, 1] >= min_timesteps
            pair_len_mask =  prim_pairs[:, 3] - prim_pairs[:, 2] >= min_timesteps
            prim_singles = prim_singles[single_len_mask]
            prim_pairs = prim_pairs[pair_len_mask]
            
            single_mask = prim_singles[:, 0] == agent_idx
            pair_mask = (prim_pairs[:, 0] == agent_idx) | (prim_pairs[:, 1] == agent_idx)

            agent_singles = prim_singles[single_mask]
            agent_pairs = prim_pairs[pair_mask]
            cluster_singles = clusters_single[single_mask]
            cluster_pairs = clusters_pair[pair_mask]

            combined_single = np.concatenate([agent_singles, cluster_singles], axis=-1)
            combined_pair = np.concatenate([agent_pairs, cluster_pairs], axis=-1)
            single_lens = combined_single[:, 2] - combined_single[:, 1]
            pair_lens = combined_pair[:, 3] - combined_pair[:, 2]
            if not len(agent_singles):
                single_dist = 0
            else:
                single_dist = (combined_single[:, -1]*single_lens).sum()/single_lens.sum()
                # Length of flattened single primitive
                single_dist /= np.sqrt(303)
            
            if not len(agent_pairs):
                pair_dists = np.array([])
                pair_ids = np.array([])
            else:
                unique_others, other_cnts = np.unique(combined_pair[:, 1], return_counts=True)
                pair_dists = np.array([(combined_pair[(idxs := combined_pair[:, 1] == other)][:, -1]*pair_lens[idxs]).sum()/(pair_lens[idxs].sum()) for other in unique_others])
                # Length of flattened pair primitive
                pair_dists /= np.sqrt(606)
                # TODO: also output the ID of the corresponding pairs here...
                pair_ids = unique_others.astype(int)

            single_outs.append(single_dist)
            pair_outs.append(pair_dists)
            pair_id_outs.append(pair_ids)
        return single_outs, pair_outs, pair_id_outs

    for split in ['training', 'validation', 'testing']:
        for shard_idx in range(shards):
            loaded = load_base_with_prims_clusters(split, base_path, cache_path, shards, num_scenarios, 
                                                   hist_only, extrap, shard_idx=shard_idx)

            if args.parallel:
                from joblib import Parallel, delayed    
                outs = Parallel(n_jobs=args.nproc, batch_size=4)(delayed(process)(*input)
                    for input in tqdm(zip(*loaded), 'Processing shard', total=len(loaded[0]), dynamic_ncols=True))
            else:
                outs = []
                for input in tqdm(zip(*loaded), 'Processing shard', total=len(loaded[0]), dynamic_ncols=True):
                    out = process(*input)
                    outs.append(out)

            outs = [{'singles': np.array(x[0]), 'pairs': np.array(x[1]), 'pair_ids': np.array(x[2])} for x in outs]

            CACHE_SUBDIR = os.path.join(CACHE_DIR,split, 'cluster_anomaly')
            os.makedirs(CACHE_SUBDIR, exist_ok=True)
            hist_suffix = '_hist' if hist_only else '_extrap' if extrap else ''
            shard_suffix = f'_shard{shard_idx}_{shards}' if shards > 1 else ''
            out_path = f'{CACHE_SUBDIR}/anomaly{hist_suffix}{shard_suffix}.npz'

            with open(out_path, 'wb') as f:
                np.savez_compressed(f, outs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='/av_shared_ssd/mtr_process_ssd')
    parser.add_argument('--cache_path', type=str, default='/av_shared_ssd/scenario_id/cache')
    parser.add_argument('--num_scenarios', type=int, default=-1)
    parser.add_argument('--cfg_file', default='../cfgs/meta/mtr+20p_mini.yaml', help='Which ensemble config to use')
    parser.add_argument('--extra_tag', default='default')
    parser.add_argument('--split', default='training', choices=['training', 'validation', 'testing'])
    parser.add_argument('--shards', type=int, default=10)
    parser.add_argument('--num_shards', type=int, default=-1)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--hist_only', action='store_true')
    parser.add_argument('--min_timesteps', type=int, default=5, help='Minimum length of a primitive before normalizing')
    parser.add_argument('--max_add', type=int, default=100000)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--extrap', action='store_true')
    parser.add_argument('--nproc', type=int, default=20)
    parser.add_argument('--load_dists', action='store_true', help='Load distances')
    parser.add_argument('--load_labels', action='store_true', help='Load cluster labels')
    args = parser.parse_args()

    assert not (args.extrap and args.hist_only), 'Only one of extrap and hist_only permitted'

    # Takes ~1-2 minutes
    calc_anomaly(args, args.base_path, args.cache_path, args.num_scenarios,
                args.shards, args.num_shards, args.hist_only, args.extrap, args.min_timesteps, args.load_dists)
    
