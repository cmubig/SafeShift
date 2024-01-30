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

VISUAL_OUT_SUBDIR = os.path.join(VISUAL_OUT_DIR, f"{os.path.basename(__file__).split('.')[0]}")

def get_velocities(interp_vals):
    vels = np.zeros_like(interp_vals[:, :3])
    vels[1:] = (interp_vals[1:, :3] - interp_vals[:-1, :3])*10
    vels[0] = vels[1]
    return vels

def get_singles(interp_vals, full_valid=False, interp_only=True):
    singles = []
    indices = []
    speeds = [np.linalg.norm(get_velocities(x), axis=-1)[:, np.newaxis] for x in interp_vals]
    for n, speed1 in zip(range(len(interp_vals)), speeds):
        interp_val1 = interp_vals[n]
        if full_valid and not (interp_val1[:, -1] > 0).all():
            continue
        if interp_only:
            valid_idxs = np.where(interp_val1[:, -1] > 0)[0]
            if not len(valid_idxs):
                continue
            range_start, range_end = valid_idxs[0], valid_idxs[-1] + 1
        else:
            range_start, range_end = 0, len(interp_val1)
        if range_end - range_start < 1:
            continue
        xyz1 = interp_val1[:, :3]
        xyz1_ = xyz1[range_start:range_end]
        speed1_ = speed1[range_start:range_end]
        y_single = np.concatenate([xyz1_, speed1_], axis=-1)
        assert range_start >= 0 and range_end <= len(interp_val1), 'Range start and end outside of possibility'
        if np.isnan(y_single).any():
            continue
        singles.append(y_single)
        indices.append(tuple([n, range_start, range_end]))
    return singles, indices

def get_encounters(interp_vals, full_valid=False, interp_only=True):
    encounters = []
    indices = []
    speeds = [np.linalg.norm(get_velocities(x), axis=-1)[:, np.newaxis] for x in interp_vals]
    for n, speed1 in zip(range(len(interp_vals)), speeds):
        for n2, speed2 in zip(range(len(interp_vals)), speeds):
            if n2 <= n:
                continue
            interp_val1 = interp_vals[n]
            interp_val2 = interp_vals[n2]
            if full_valid and not (interp_val1[:, -1] > 0).all():
                continue
            if full_valid and not (interp_val2[:, -1] > 0).all():
                continue
            dists = np.linalg.norm(interp_val1[:, :3] - \
                                    interp_val2[:, :3], axis=-1)
            if dists.max() >= 100:
                continue
            interp_val1 = interp_vals[n]
            interp_val2 = interp_vals[n2]
            if interp_only:
                valid_idxs1 = np.where(interp_val1[:, -1] > 0)[0]
                if not len(valid_idxs1):
                    continue
                valid_idxs2 = np.where(interp_val2[:, -1] > 0)[0]
                if not len(valid_idxs2):
                    continue
                range_start, range_end = max(valid_idxs1[0], valid_idxs2[0]), min(valid_idxs1[-1] + 1, valid_idxs2[-1] + 1)
            else:
                range_start, range_end = 0, len(interp_val1)
            if range_end - range_start < 1:
                continue
            assert range_start >= 0 and range_end <= len(interp_val1), 'Range start and end outside of possibility'

            xyz1 = interp_val1[:, :3]
            xyz2 = interp_val2[:, :3]

            xyz1_ = xyz1[range_start:range_end]
            speed1_ = speed1[range_start:range_end]
            xyz2_ = xyz2[range_start:range_end]
            speed2_ = speed2[range_start:range_end]
            y_pair = np.concatenate([xyz1_, xyz2_, speed1_, speed2_], axis=-1)

            if np.isnan(y_pair).any():
                continue
            encounters.append(y_pair)
            indices.append((n, n2, range_start, range_end))
    return encounters, indices

def load_base(split, base_path, cache_path, shards, num_scenarios, hist_only, shard_idx=0):
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

    return metas, inputs, interp_vals

def build_hdps(input, interp_vals, n_added_single, n_added_pair, single_model, pair_model, hist_only=False, extrap=False, max_add=20000):
    if hist_only:
        interp_vals = interp_vals[:, :11]

    if n_added_single < max_add:
        for y_single, agent_idxs in zip(*get_singles(interp_vals, full_valid=(not extrap), interp_only=(not extrap))):
            if n_added_single < max_add:
                if np.isnan(y_single).any():
                    continue
                single_model.add_data(y_single)
                n_added_single += 1
    single_added = n_added_single < max_add

    if n_added_pair < max_add:
        for y_pair, agent_idxs in zip(*get_encounters(interp_vals, full_valid=(not extrap), interp_only=(not extrap))):
            if n_added_pair < max_add:
                if np.isnan(y_pair).any():
                    continue
                pair_model.add_data(y_pair)
                n_added_pair += 1
    pair_added = n_added_pair < max_add

    return (single_added or pair_added), n_added_single, n_added_pair

def get_posterior_models(args, base_path, cache_path, shards, num_scenarios, hist_only, extrap, max_add, single_path, pair_path):
    if not args.load_model:
        obs_dim_single = 4
        obs_hypparams_single = {'mu_0':np.zeros(obs_dim_single),
                        'sigma_0':np.eye(obs_dim_single),
                        'kappa_0':0.25,
                        'nu_0':obs_dim_single+2}
        obs_dim_pair = 8
        obs_hypparams_pair = {'mu_0':np.zeros(obs_dim_pair),
                        'sigma_0':np.eye(obs_dim_pair),
                        'kappa_0':0.25,
                        'nu_0':obs_dim_pair+2}
        Nmax = 100

        obs_distns_single = [pyhsmm.distributions.Gaussian(**obs_hypparams_single) for state in range(Nmax)]
        posteriormodel_single = pyhsmm.models.WeakLimitStickyHDPHMM(kappa=5, alpha=2.,gamma=5,
                                                            init_state_concentration=1., obs_distns=obs_distns_single)
        n_added_single = 0
        obs_distns_pair = [pyhsmm.distributions.Gaussian(**obs_hypparams_pair) for state in range(Nmax)]
        posteriormodel_pair = pyhsmm.models.WeakLimitStickyHDPHMM(kappa=5, alpha=2.,gamma=5,
                                                            init_state_concentration=1., obs_distns=obs_distns_pair)
        n_added_pair = 0
        metas, inputs, interp_vals = load_base('training', base_path, cache_path, shards, num_scenarios, hist_only or extrap, shard_idx=0)
        for _, input, interp_val in tqdm(zip(metas, inputs, interp_vals), 'Processing scenarios', total=len(interp_vals)):
            added, new_added_single, new_added_pair = build_hdps(input, interp_val, n_added_single, n_added_pair,
                                                                 posteriormodel_single, posteriormodel_pair,
                                                                 hist_only=hist_only, extrap=extrap, max_add=max_add)
            n_added_single = new_added_single
            n_added_pair = new_added_pair
            if not added:
                break

        for _ in tqdm(range(100), 'Training BNP models'):
            posteriormodel_pair.resample_model(num_procs=20)
            posteriormodel_single.resample_model(num_procs=20)
        with open(single_path, 'wb') as f:
            pkl.dump(posteriormodel_single, f)
        with open(pair_path, 'wb') as f:
            pkl.dump(posteriormodel_pair, f)
    else:
        with open(single_path, 'rb') as f:
            posteriormodel_single = pkl.load(f)
        with open(pair_path, 'rb') as f:
            posteriormodel_pair = pkl.load(f)
    return posteriormodel_single, posteriormodel_pair

def assign_primitives(input, interp_vals, model_single, model_pair, single_path, pair_path, hist_only=False, extrap=False):
    if hist_only:
        interp_vals = interp_vals[:, :11]
    
    if model_single is None:
        with open(single_path, 'rb') as f:
            model_single = pkl.load(f)
    if model_pair is None:
        with open(pair_path, 'rb') as f:
            model_pair = pkl.load(f)

    # 3D: index start, index end, primitive ID
    def change_point_ids(stateseq):
        change_points = [x for x in pyhsmm.util.general.labels_to_changepoints(stateseq)]
        state_vals = [stateseq[x[0]] for x in change_points]
        return np.concatenate([np.array(change_points), np.array(state_vals)[:, np.newaxis]], axis=-1).astype(np.float32)
        
    singles = []
    single_idxs = []
    for y_single, agent_idxs in zip(*get_singles(interp_vals, interp_only=(not extrap))):
        assert not np.isnan(y_single).any(), 'No nan allowed here'
        model_single.add_data(y_single)
        stateseq = model_single.states_list[-1].stateseq.copy()
        model_single.states_list.pop()

        single_idxs.append(np.array(agent_idxs))
        singles.append(change_point_ids(stateseq))
    single_idxs = np.array(single_idxs)
    # Final output = n_encounters x 4:
    #  (agent_idx, primitive_start_idx, primitive_end_idx [non-inclusive], primitive_id)
    single_out = []
    for idxs, data in zip(single_idxs, singles):
        to_extend = []
        agent_id, range_start, _ = idxs
        for primitive in data:
            primitive_start, primitive_end, primitive_id = primitive
            primitive_start += range_start
            primitive_end += range_start
            to_extend.append([agent_id, primitive_start, primitive_end, primitive_id])
        single_out.extend(to_extend)
    single_out = np.array(single_out)

    pairs = []
    pair_idxs = []
    for y_pair, agent_idxs in zip(*get_encounters(interp_vals, interp_only=(not extrap))):
        assert not np.isnan(y_pair).any(), 'No nan allowed here'
        model_pair.add_data(y_pair)
        stateseq = model_pair.states_list[-1].stateseq.copy()
        model_pair.states_list.pop()

        pair_idxs.append(np.array(agent_idxs))
        pairs.append(change_point_ids(stateseq))
    pair_idxs = np.array(pair_idxs)
    # Final output = n_encounters x 5:
    #  (agent_idx1, agent_idx2, primitive_start_idx, primitive_end_idx [non-inclusive], primitive_id)
    pair_out = []
    for idxs, data in zip(pair_idxs, pairs):
        to_extend = []
        agent_id1, agent_id2, range_start, _ = idxs
        for primitive in data:
            primitive_start, primitive_end, primitive_id = primitive
            primitive_start += range_start
            primitive_end += range_start
            ids_sorted = sorted([agent_id1, agent_id2])
            to_extend.append([*ids_sorted, primitive_start, primitive_end, primitive_id])
        pair_out.extend(to_extend)
    pair_out = np.array(pair_out)

    return {'singles': single_out, 'pairs': pair_out}

def extract_primitives(args, base_path, cache_path, num_scenarios=-1, shards=10, num_shards=-1, hist_only=False, extrap=False):
    splits = ['training', 'validation', 'testing']

    # Model output path
    CACHE_SUBDIR = os.path.join(CACHE_DIR, 'training', 'primitives')
    os.makedirs(CACHE_SUBDIR, exist_ok=True)
    hist_suffix = '_hist' if hist_only else '_extrap' if extrap else ''
    single_path = f'{CACHE_SUBDIR}/posterior_single{hist_suffix}.pkl' 
    pair_path = f'{CACHE_SUBDIR}/posterior_pair{hist_suffix}.pkl' 
    max_add = 20000
    posteriormodel_single, posteriormodel_pair = get_posterior_models(args, base_path, cache_path, shards,
                                                                      num_scenarios, hist_only, extrap, max_add,
                                                                      single_path, pair_path)

    states = posteriormodel_single.states_list
    tot_unique = sum([len(np.unique(states[i].stateseq)) for i in range(len(states))])
    tot_prims = set()
    for x in states:
        tot_prims.update(x.stateseq)
    print('Single', len(tot_prims), tot_unique)
    states = posteriormodel_pair.states_list
    tot_unique = sum([len(np.unique(states[i].stateseq)) for i in range(len(states))])
    tot_prims = set()
    for x in states:
        tot_prims.update(x.stateseq)
    print('Pair', len(tot_prims), tot_unique)

    # Now, we actually output the labeled primitives
    for split in splits:
        CACHE_SUBDIR = os.path.join(CACHE_DIR, split, 'primitives')
        os.makedirs(CACHE_SUBDIR, exist_ok=True)
        for shard_idx in range(10):
            metas, inputs, interp_vals = load_base(split, base_path, cache_path, shards, num_scenarios, \
                                                   hist_only or extrap, shard_idx=shard_idx)
            msg = f'Processing scenarios for {split}, shard {shard_idx}'

            if args.parallel:
                from joblib import Parallel, delayed    
                all_outs = Parallel(n_jobs=args.nproc, batch_size=4)(delayed(assign_primitives)(
                    input, interp_val, None, None,
                    single_path=single_path, pair_path=pair_path, hist_only=hist_only, extrap=extrap)
                    for _, input, interp_val in tqdm(zip(metas, inputs, interp_vals), msg, total=len(metas)))
            else:
                all_outs = []
                for _, input, interp_val in tqdm(zip(metas, inputs, interp_vals), msg, total=len(metas)):
                    out = assign_primitives(input, interp_val, None, None,
                                            single_path=single_path, pair_path=pair_path, hist_only=hist_only, extrap=extrap)
                    all_outs.append(out)

            shard_suffix = f'_shard{shard_idx}_{10}'
            with open(f'{CACHE_SUBDIR}/prims{hist_suffix}{shard_suffix}.npz', 'wb') as f:
                np.savez_compressed(f, all_outs)

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
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--extrap', action='store_true')
    parser.add_argument('--nproc', type=int, default=20)
    parser.add_argument('--recache', action='store_true', help='Ignore/rebuild cache')
    args = parser.parse_args()

    assert not (args.extrap and args.hist_only), 'Only one of extrap and hist_only permitted'

    extract_primitives(args, args.base_path, args.cache_path, args.num_scenarios,
                           args.shards, args.num_shards, args.hist_only, args.extrap)
