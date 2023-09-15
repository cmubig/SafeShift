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
    POS_XYZ_IDX, CACHE_DIR, VISUAL_OUT_DIR, POS_XY_IDX, VEL_XY_IDX, HEADING_IDX 
)
from primitives import get_encounters, get_singles, get_velocities

VISUAL_OUT_SUBDIR = os.path.join(VISUAL_OUT_DIR, f"{os.path.basename(__file__).split('.')[0]}")

def load_base_with_prims(split, base_path, cache_path, shards, num_scenarios, hist_only, extrap, shard_idx=0):
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
    file_name = 'interp' if (not hist_only and not extrap) else 'interp_hist'
    shard_suffix = f'_shard{shard_idx}_{shards}' if shards > 1 else ''
    interp_vals_filepath = os.path.join(cache_path, f"{split}/frenet/{file_name}{shard_suffix}.npz")
    interp_vals = np.load(interp_vals_filepath, allow_pickle=True)['arr_0']
    print(f"Loading shards took {time.time() - start} seconds")

    print(f"Loading primitive cache {shard_idx} of {10}")
    file_name = 'prims_hist' if hist_only else 'prims_extrap' if extrap else 'prims'
    shard_suffix = f'_shard{shard_idx}_{shards}' if shards > 1 else ''
    primitives_filepath = os.path.join(cache_path, f"{split}/primitives_spline/{file_name}{shard_suffix}.npz")
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

def normalize_time(xyz, step=0.01):
    assert len(xyz) > 1, 'At least one data point needed to perform linear interpolation'
    timesteps = np.arange(len(xyz))
    timesteps = timesteps / timesteps[-1]
    new_times = np.arange(0, 1+step, step)
    xyz_norm = np.stack([np.interp(new_times, timesteps, xyz[:, i]) for i in range(xyz.shape[1])], axis=-1)
    return xyz_norm

def center_traj(xyz):
    return xyz - xyz.mean(axis=0)

def optimally_align_singles(a, b):
    """ Computes optimal alignment
        Returns rmsd: float,
                rot: Rotation [to apply to center_traj(b)]
    """
    assert len(set([len(x) for x in [a, b]])) == 1, 'All vectors must have the same length'
    a = center_traj(a)
    b = center_traj(b)
    rot, rssd = Rotation.align_vectors(a, b)
    return rssd / np.sqrt(len(a)), rot

def optimally_align_and_order_pairs(a, b, c, d):
    """ Computes optimal alignment and ordering. 
        center_traj(a cat b) to center_traj(c cat d), as well as 
        center_traj(a cat b) to center_traj(d cat c) are tested.
        Returns rmsd: float,
                rot: Rotation [to apply to center_traj(c cat d) or center_traj(d cat c)],
                swap_cd: bool [whether or not to swap c and d in order]
    """
    assert len(set([len(x) for x in [a, b, c, d]])) == 1, 'All vectors must have the same length'
    ab = center_traj(np.concatenate([a, b], axis=0))
    cd = center_traj(np.concatenate([c, d], axis=0))
    dc = center_traj(np.concatenate([d, c], axis=0))
    rot_cd, rssd_cd = Rotation.align_vectors(ab, cd)
    rot_dc, rssd_dc = Rotation.align_vectors(ab, dc)
    rmsd_cd = rssd_cd / np.sqrt(len(a))
    rmsd_dc = rssd_dc / np.sqrt(len(a))

    if rmsd_cd <= rmsd_dc:
        return rmsd_cd, rot_cd, False
    else:
        return rmsd_dc, rot_dc, True

def rmsd(a, b):
    distances = np.linalg.norm(a - b, axis=-1)
    rmsd = np.sqrt((distances ** 2).sum() / len(distances))
    return rmsd

def norm_encounters(input, interp_val, primitive, max_add, n_added_single, n_added_pair, hist_only=False, min_timesteps=5):
    # First geometric approximation: orient everything to the first element, then do k-means with aligned L2 dist.
    single_out = []
    for single_info in primitive['singles']:
        if n_added_single >= max_add:
            break
        agent_idx, start, end, prim_id = single_info.astype(int)
        if end - start < min_timesteps:
            continue
        n_added_single += 1
        xyz = interp_val[agent_idx, start:end, :3]
        xyz_norm = normalize_time(xyz)
        single_out.append(xyz_norm)
    single_out = np.array(single_out)
    # N_s x 1 x 101 x 3
    single_out = single_out[:, np.newaxis]

    pair_out = []
    for pair_info in primitive['pairs']:
        if n_added_pair >= max_add:
            break
        agent_idx1, agent_idx2, start, end, prim_id = pair_info.astype(int)
        if end - start < min_timesteps:
            continue
        n_added_pair += 1
        xyz1 = interp_val[agent_idx1, start:end, :3]
        xyz_norm1 = normalize_time(xyz1)
        xyz2 = interp_val[agent_idx2, start:end, :3]
        xyz_norm2 = normalize_time(xyz2)
        pair_out.append([xyz_norm1, xyz_norm2])
    # N_p x 2 x 101 x 3
    pair_out = np.array(pair_out)

    return n_added_single, n_added_pair, single_out, pair_out

def get_aligns(all_singles, all_pairs, single_base1=None, pair_base1=None, pair_base2=None, do_tmux=True):
    if single_base1 is None:
        single_base1 = all_singles[0][0]
    if pair_base1 is None or pair_base2 is None:
        pair_base1, pair_base2 = all_pairs[0][0], all_pairs[0][1]

    single_aligned = []
    for single in tqdm(all_singles, 'Aligning singles', disable=(not do_tmux)):
        xyz1 = single[0]
        _, rot = optimally_align_singles(single_base1, xyz1)
        aligned1 = rot.apply(center_traj(xyz1))
        single_aligned.append(aligned1)

    pair_aligned = []
    for pair in tqdm(all_pairs, 'Aligning pairs', disable=(not do_tmux)):
        xyz1, xyz2 = pair[0], pair[1]
        _, rot, swap = optimally_align_and_order_pairs(pair_base1, pair_base2, xyz1, xyz2)
        aligned12 = rot.apply(center_traj(np.concatenate(
            [xyz1 if not swap else xyz2, xyz2 if not swap else xyz1], axis=0)))
        pair_aligned.append(aligned12)
    return single_aligned, pair_aligned

def get_distmats(args, base_path, cache_path, num_scenarios=-1, shards=10, num_shards=-1,
                  hist_only=False, extrap=False, min_timesteps=5, load_dists=False):
    CACHE_SUBDIR = os.path.join(CACHE_DIR, 'training', 'cluster')
    os.makedirs(CACHE_SUBDIR, exist_ok=True)
    hist_suffix = '_hist' if hist_only else '_extrap' if extrap else ''
    single_path = f'{CACHE_SUBDIR}/single_aligns{hist_suffix}.npz' 
    pair_path = f'{CACHE_SUBDIR}/pair_aligns{hist_suffix}.npz' 

    if load_dists:
        print('Loading dist matrices from disk')
        start = time.time()
        single_dict = np.load(single_path, allow_pickle=True)['arr_0']
        pair_dict = np.load(pair_path, allow_pickle=True)['arr_0']
        print(f'Process took {time.time() - start}')
        single_dict = single_dict.item()
        pair_dict = pair_dict.item()
        return single_dict['entries'], pair_dict['entries'], single_dict['aligns'], pair_dict['aligns']
    
    # Max for performing the actual clustering, since intractable to go much beyond that with N^3 algorithms
    max_add = args.max_add
    n_added_single = 0
    n_added_pair = 0
    metas, inputs, interp_vals, primitives = load_base_with_prims('training', base_path, cache_path, shards, num_scenarios, hist_only, extrap, shard_idx=0)
    all_singles = []
    all_pairs = []
    for _, input, interp_val, primitive in tqdm(zip(metas, inputs, interp_vals, primitives), 'Processing scenarios', total=len(interp_vals)):
        new_single, new_pair, single_out, pair_out = norm_encounters(input, interp_val, primitive, max_add,
                                                                     n_added_single, n_added_pair,
                                                                     hist_only=hist_only, min_timesteps=min_timesteps)
        if len(single_out):
            all_singles.append(single_out)
        if len(pair_out):
            all_pairs.append(pair_out)

        n_added_single = new_single
        n_added_pair = new_pair
        if n_added_single >= max_add and n_added_pair >= max_add:
            break
    all_singles = np.concatenate(all_singles, axis=0)
    all_pairs = np.concatenate(all_pairs, axis=0)

    single_aligns, pair_aligns = get_aligns(all_singles, all_pairs)

    # Model output path
    print('Saving dist matrices to disk')
    with open(single_path, 'wb') as f:
        np.savez_compressed(f, {'aligns': single_aligns, 'entries': all_singles})
    with open(pair_path, 'wb') as f:
        np.savez_compressed(f, {'aligns': pair_aligns, 'entries': all_pairs})

    return all_singles, all_pairs, single_aligns, pair_aligns

def do_clustering(args, base_path, cache_path, num_scenarios=-1, shards=10, num_shards=-1,
                  hist_only=False, extrap=False, min_timesteps=5, load_dists=False):

    all_singles, all_pairs, single_aligns, pair_aligns = get_distmats(args, base_path, cache_path, num_scenarios, shards,
                                                                    num_shards, hist_only, extrap, min_timesteps, load_dists)
    single_data = np.array([x.flatten() for x in single_aligns])
    pair_data = np.array([x.flatten() for x in pair_aligns])

    print('Clustering...')
    start = time.time()
    k = 20
    kmeans_single = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans_single = kmeans_single.fit(single_data)
    kmeans_pair = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans_pair = kmeans_pair.fit(pair_data)
    print(f'Done in {time.time() - start}')

    CACHE_SUBDIR = os.path.join(CACHE_DIR, 'training', 'cluster')
    os.makedirs(CACHE_SUBDIR, exist_ok=True)
    hist_suffix = '_hist' if hist_only else '_extrap' if extrap else ''
    single_path = f'{CACHE_SUBDIR}/single_kmeans{hist_suffix}.npz' 
    pair_path = f'{CACHE_SUBDIR}/pair_kmeans{hist_suffix}.npz' 
    with open(single_path, 'wb') as f:
        np.savez_compressed(f, {'labels': kmeans_single.labels_, 'centers': kmeans_single.cluster_centers_})
    with open(pair_path, 'wb') as f:
        np.savez_compressed(f, {'labels': kmeans_pair.labels_, 'centers': kmeans_pair.cluster_centers_})

    return kmeans_single, kmeans_pair

def label_others(args, base_path, cache_path, shards=10, num_scenarios=-1, hist_only=False, extrap=False, min_timesteps=5):
    splits = ['training', 'validation', 'testing']
    # Now, we actually output the labeled primitives

    CACHE_SUBDIR = os.path.join(CACHE_DIR, 'training', 'cluster')
    os.makedirs(CACHE_SUBDIR, exist_ok=True)
    hist_suffix = '_hist' if hist_only else '_extrap' if extrap else ''
    kmeans_single_path = f'{CACHE_SUBDIR}/single_kmeans{hist_suffix}.npz' 
    kmeans_pair_path = f'{CACHE_SUBDIR}/pair_kmeans{hist_suffix}.npz' 
    kmeans_single = np.load(kmeans_single_path, allow_pickle=True)['arr_0'].item()
    kmeans_pair = np.load(kmeans_pair_path, allow_pickle=True)['arr_0'].item()
    align_single_path = f'{CACHE_SUBDIR}/single_aligns{hist_suffix}.npz' 
    align_pair_path = f'{CACHE_SUBDIR}/pair_aligns{hist_suffix}.npz' 
    align_single = np.load(align_single_path, allow_pickle=True)['arr_0'].item()
    align_pair = np.load(align_pair_path, allow_pickle=True)['arr_0'].item()

    single_base1 = align_single['entries'][0][0]
    pair_base1, pair_base2 = align_pair['entries'][0][0], align_pair['entries'][0][1]

    for split in splits:
        CACHE_SUBDIR = os.path.join(CACHE_DIR, split, 'cluster')
        os.makedirs(CACHE_SUBDIR, exist_ok=True)
        for shard_idx in range(10):
            metas, inputs, interp_vals, primitives = load_base_with_prims(split, base_path, cache_path, shards, num_scenarios, hist_only, extrap, shard_idx=shard_idx)
            msg = f'Processing scenarios for {split}, shard {shard_idx}'

            def process(input, interp_val, primitive):
                _, _, singles, pairs = norm_encounters(input, interp_val, primitive, 1e100, 0, 0, hist_only, min_timesteps)
                single_aligns, pair_aligns = get_aligns(singles, pairs, single_base1, pair_base1, pair_base2, do_tmux=False)
                single_data = np.array([x.flatten() for x in single_aligns])
                pair_data = np.array([x.flatten() for x in pair_aligns])
                if len(single_data):
                    single_dists = np.linalg.norm(kmeans_single['centers'] - single_data[:, np.newaxis, :], axis=-1) 
                    single_assignments = single_dists.argmin(axis=-1)
                    single_min_dists = single_dists.min(axis=-1)
                    single_out = np.stack([single_assignments, single_min_dists], axis=-1)
                else:
                    single_out = np.zeros((0, 2))

                if len(pair_data):
                    pair_dists = np.linalg.norm(kmeans_pair['centers'] - pair_data[:, np.newaxis, :], axis=-1) 
                    pair_assignments = pair_dists.argmin(axis=-1)
                    pair_min_dists = pair_dists.min(axis=-1)
                    pair_out = np.stack([pair_assignments, pair_min_dists], axis=-1)
                else:
                    pair_out = np.zeros((0, 2))

                return single_out, pair_out

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

            single_outs = [x[0] for x in all_outs]
            pair_outs = [x[1] for x in all_outs]
            shard_suffix = f'_shard{shard_idx}_{10}'
            with open(f'{CACHE_SUBDIR}/single_center_dists{hist_suffix}{shard_suffix}.npz', 'wb') as f:
                np.savez_compressed(f, single_outs)
            with open(f'{CACHE_SUBDIR}/pair_center_dists{hist_suffix}{shard_suffix}.npz', 'wb') as f:
                np.savez_compressed(f, pair_outs)

def visualize_clusters(args, base_path, cache_path, num_scenarios=-1, shards=10, hist_only=False, extrap=False):
    CACHE_SUBDIR = os.path.join(CACHE_DIR, 'training', 'cluster')
    os.makedirs(CACHE_SUBDIR, exist_ok=True)
    VIS_SUBDIR = os.path.join('out/scenario_id/vis', 'cluster')
    os.makedirs(VIS_SUBDIR, exist_ok=True)

    hist_suffix = '_hist' if hist_only else '_extrap' if extrap else ''
    kmeans_single_path = f'{CACHE_SUBDIR}/single_kmeans{hist_suffix}.npz' 
    kmeans_pair_path = f'{CACHE_SUBDIR}/pair_kmeans{hist_suffix}.npz' 
    kmeans_single = np.load(kmeans_single_path, allow_pickle=True)['arr_0'].item()
    kmeans_pair = np.load(kmeans_pair_path, allow_pickle=True)['arr_0'].item()

    n_single = np.unique(kmeans_single['labels'], return_counts=True)[-1]
    n_pair = np.unique(kmeans_pair['labels'], return_counts=True)[-1]
    for i, (center, n) in enumerate(zip(kmeans_single['centers'], n_single)):
        folder = os.path.join(VIS_SUBDIR, f'single{hist_suffix}')
        os.makedirs(folder, exist_ok=True)

        plt.clf()
        center = center.reshape(-1, 3)
        plt.plot(center[:, 0], center[:, 1], marker='.')
        plt.title(f'cluster single{hist_suffix} {i}, N={n}')
        plt.savefig(os.path.join(folder, f'cluster_{i}.png'), dpi=300)

    for i, (center, n) in enumerate(zip(kmeans_pair['centers'], n_pair)):
        folder = os.path.join(VIS_SUBDIR, f'pair{hist_suffix}')
        os.makedirs(folder, exist_ok=True)

        plt.clf()
        center = center.reshape(-1, 3)
        center1 = center[:int(len(center)/2)]
        center2 = center[int(len(center)/2):]
        plt.plot(center1[:, 0], center1[:, 1], marker='.')
        plt.plot(center2[:, 0], center2[:, 1], marker='.')
        plt.title(f'cluster pair{hist_suffix} {i}, N={n}')
        plt.savefig(os.path.join(folder, f'cluster_{i}.png'), dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='~/monet_shared/shared/mtr_process')
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
    if not args.load_labels:
        do_clustering(args, args.base_path, args.cache_path, args.num_scenarios,
                    args.shards, args.num_shards, args.hist_only, args.extrap, args.min_timesteps, args.load_dists)
        label_others(args, args.base_path, args.cache_path, args.shards, args.num_scenarios,
                    args.hist_only, args.extrap, min_timesteps=args.min_timesteps)
    
    visualize_clusters(args, args.base_path, args.cache_path, args.shards, args.num_scenarios, args.hist_only, args.extrap)
    pass
    
    
