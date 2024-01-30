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

from mtr.datasets.waymo.waymo_eval import waymo_evaluation_explore, waymo_evaluation
from mtr.utils.motion_utils import batch_nms
from mtr.config import cfg, cfg_from_yaml_file
from mtr.utils import common_utils

from tools.scenario_identification.utils.common import (
    plot_static_map_infos, plot_dynamic_map_infos, plot_lanes_by_distance,
    POS_XYZ_IDX, CACHE_DIR, VISUAL_OUT_DIR, POS_XY_IDX, VEL_XY_IDX, HEADING_IDX 
)
from primitives import get_encounters, get_singles, get_velocities

VISUAL_OUT_SUBDIR = os.path.join(VISUAL_OUT_DIR, f"{os.path.basename(__file__).split('.')[0]}")

def load_base_with_prims(split, base_path, cache_path, shards, num_scenarios, hist_only, shard_idx=0):
    base = os.path.expanduser(base_path)
    step = 1
    if split == 'training':
        step = 5
        scenarios_base = f'{base}/new_processed_scenarios_training'
        scenarios_meta = f'{base}/new_processed_scenarios_training_infos.pkl'
    elif split == 'validation':
        scenarios_base = f'{base}/new_processed_scenarios_validation'
        scenarios_meta = f'{base}/new_processed_scenarios_val_infos.pkl'
    else:
        scenarios_base = f'{base}/new_processed_scenarios_testing'
        scenarios_meta = f'{base}/new_processed_scenarios_test_infos.pkl'

    start = time.time()
    print(f"Loading {split} Scenario Data...")
    with open(scenarios_meta, 'rb') as f:
        metas = pkl.load(f)[::step]
    inputs = [(f'sample_{x["scenario_id"]}.pkl', f'{scenarios_base}/sample_{x["scenario_id"]}.pkl') for x in metas]
    print(f"Process took {time.time() - start} seconds.")

    start = time.time()
    print(f"Loading cache {shard_idx} of {10}")
    file_name = 'interp' if not hist_only else 'interp_hist'
    shard_suffix = f'_shard{shard_idx}_{shards}' if shards > 1 else ''
    interp_vals_filepath = os.path.join(cache_path, f"{split}/frenet/{file_name}{shard_suffix}.npz")
    interp_vals = np.load(interp_vals_filepath, allow_pickle=True)['arr_0']
    print(f"Loading shards took {time.time() - start} seconds")

    print(f"Loading primitive cache {shard_idx} of {10}")
    file_name = 'prims' if not hist_only else 'prims_hist'
    shard_suffix = f'_shard{shard_idx}_{shards}' if shards > 1 else ''
    primitives_filepath = os.path.join(cache_path, f"{split}/primitives/{file_name}{shard_suffix}.npz")
    primitives = np.load(primitives_filepath, allow_pickle=True)['arr_0']
    print(f"Loading primitive shards took {time.time() - start} seconds")

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

    return metas, inputs, interp_vals, primitives

def process_prims(args, base_path, cache_path, shards=10, num_shards=10, num_scenarios=-1, hist_only=False, min_timesteps=5):
    split = 'validation'
    CACHE_SUBDIR = os.path.join(CACHE_DIR, split, 'cluster')
    os.makedirs(CACHE_SUBDIR, exist_ok=True)
    for shard_idx in range(num_shards):
        metas, inputs, interp_vals, primitives = load_base_with_prims(split, base_path, cache_path, shards, num_scenarios, hist_only, shard_idx=shard_idx)
        msg = f'Processing scenarios for {split}, shard {shard_idx}'

        def process(input, interp_val, primitive):
            singles = primitive['singles']
            single_ids = singles[:, -1]
            single_lengths = singles[:, -2] - singles[:, -3]
            single_n_prims, single_n_prim_cnts = np.unique(np.unique(singles[:, 0], return_counts=True)[1], return_counts=True)

            pairs = primitive['pairs']
            pair_ids = pairs[:, -1]
            pair_lengths = pairs[:, -2] - pairs[:, -3]
            pair_n_prims, pair_n_prim_cnts = np.unique(np.unique(pairs[:, :2], axis=0, return_counts=True)[1], return_counts=True)
            return single_ids, single_lengths, single_n_prims, single_n_prim_cnts, \
                   pair_ids, pair_lengths, pair_n_prims, pair_n_prim_cnts

        if args.parallel:
            from joblib import Parallel, delayed    
            all_outs = Parallel(n_jobs=args.nproc, batch_size=4)(delayed(process)(
                input, interp_val, primitive)
                for _, input, interp_val, primitive in tqdm(zip(metas, inputs, interp_vals, primitives), msg, total=len(metas)))
        else:
            all_outs = []
            for _, input, interp_val, primitive in tqdm(zip(metas, inputs, interp_vals, primitives), msg, total=len(metas)):
                out = process(input, interp_val, primitive)
                all_outs.append(out)

        single_prim_lengths = {}
        pair_prim_lengths = {}
        single_ns = {}
        pair_ns = {}
        for info in all_outs:
            single_ids, single_lengths, single_n_prims, single_n_prim_cnts, \
                pair_ids, pair_lengths, pair_n_prims, pair_n_prim_cnts = info
            for single_id, single_length in zip(single_ids, single_lengths):
                if single_id in single_prim_lengths:
                    single_prim_lengths[single_id].append(single_length)
                else:
                    single_prim_lengths[single_id] = [single_length]
            for single_n_prim, single_n_prim_cnt in zip(single_n_prims, single_n_prim_cnts):
                if single_n_prim in single_ns:
                    single_ns[single_n_prim] += single_n_prim_cnt
                else:
                    single_ns[single_n_prim] = single_n_prim_cnt
            for pair_id, pair_length in zip(pair_ids, pair_lengths):
                if pair_id in pair_prim_lengths:
                    pair_prim_lengths[pair_id].append(pair_length)
                else:
                    pair_prim_lengths[pair_id] = [pair_length]
            for pair_n_prim, pair_n_prim_cnt in zip(pair_n_prims, pair_n_prim_cnts):
                if pair_n_prim in pair_ns:
                    pair_ns[pair_n_prim] += pair_n_prim_cnt
                else:
                    pair_ns[pair_n_prim] = pair_n_prim_cnt

        single_prim_ns = {}
        pair_prim_ns = {}
        for k, v in single_prim_lengths.items():
            single_prim_lengths[k] = np.mean(v)
            single_prim_ns[k] = len(v)
        for k, v in pair_prim_lengths.items():
            pair_prim_lengths[k] = np.mean(v)
            pair_prim_ns[k] = len(v)
        breakpoint()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='~/monet_shared/shared/mtr_process')
    parser.add_argument('--cache_path', type=str, default='/av_shared_ssd/scenario_id/cache')
    parser.add_argument('--num_scenarios', type=int, default=-1)
    parser.add_argument('--cfg_file', default='../cfgs/meta/mtr+20p_mini.yaml', help='Which ensemble config to use')
    parser.add_argument('--extra_tag', default='default')
    parser.add_argument('--split', default='training', choices=['training', 'validation', 'testing'])
    parser.add_argument('--shards', type=int, default=10)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--hist_only', action='store_true')
    parser.add_argument('--min_timesteps', type=int, default=5, help='Minimum length of a primitive before normalizing')
    parser.add_argument('--max_add', type=int, default=100000)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--nproc', type=int, default=20)
    parser.add_argument('--load_dists', action='store_true', help='Load distances')
    parser.add_argument('--load_labels', action='store_true', help='Load cluster labels')
    args = parser.parse_args()

    # Takes ~1-2 minutes
    process_prims(args, args.base_path, args.cache_path, args.shards, args.num_shards, args.num_scenarios,
                args.hist_only, min_timesteps=args.min_timesteps)
    pass
    
    
