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
import contextlib
import joblib

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
from tools.scenario_identification.utils.visualization import plot_static_map_infos
from tools.scenario_identification.utils.systems import tqdm_joblib
from primitives import get_encounters, get_singles, get_velocities, load_base

VISUAL_OUT_SUBDIR = os.path.join(VISUAL_OUT_DIR, f"{os.path.basename(__file__).split('.')[0]}")

TEST_TOLS = np.array([1e-3, 0.1, 1, 2, 4, 8])


# Much of this taken from Aritra Guha's R code

def cubic_err(t, val):
    resid =  np.polyfit(t, val, deg=3, full=True)[1]
    if not len(resid):
        return 0
    return resid.item()

def cubic_residuals(t, val):
    coeffs =  np.polyfit(t, val, deg=3)
    residuals = np.polyval(coeffs, t) - val
    return residuals

# Expected dims: x_val = t, val = same shape
# start and end are inclusive, best_choice starts at last valid index
def find_poly_split(t, val, start=None, end=None, best_choice=None):
    if start is None or end is None or best_choice is None:
        start, end, best_choice = 0, len(t) - 1, len(t) - 1

    old_fit = 0
    if best_choice == len(t) - 1:
        old_fit = cubic_err(t, val)
    else:
        # Slicing is INCLUSIVE in R
        old_split1 = range(start, best_choice+1)
        old_split2 = range(best_choice, end+1)
        old_fit1 = (cubic_residuals(t[old_split1], val[old_split1])**2).sum()
        old_fit2 = (cubic_residuals(t[old_split2], val[old_split2])**2)[1:].sum()
        old_fit = old_fit1 + old_fit2
    if old_fit < 1e-16:
        return best_choice
    new_choice = int(np.floor((start + end)/2))
    new_split1 = range(0, new_choice+1)
    new_split2 = range(new_choice, len(t))
    new_fit1 = (cubic_residuals(t[new_split1], val[new_split1])**2).sum()
    new_fit2 = (cubic_residuals(t[new_split2], val[new_split2])**2)[1:].sum()
    new_fit = new_fit1 + new_fit2
    if new_fit < old_fit:
        return new_choice
    #Give up if there are only 12 observations because 
    #it's hard to fit if there are only 6
    if len(t) - start <= 11 or (end - 1) <= 11:
        return best_choice
    if abs(start - end) <= 1:
        return best_choice
    if new_fit1 < new_fit2:
        return find_poly_split(t, val, new_choice, end, best_choice)
    else:
        return find_poly_split(t, val, start, new_choice, best_choice)

# end is inclusive
def find_all_poly_splits(t, val, start=None, end=None):
    if len(t) <= 6:
        return None
    return np.arange(3, len(t) - 3, 4)
    # if start is None or end is None:
    #     start, end = 0, len(t) - 1

    # changepoint = find_poly_split(t, val)
    # if changepoint == len(t) - 1:
    #     return None
    # # If 4 or fewer from start or end
    # changepoint_idx = [x for x in range(start, end+1)][changepoint] 
    # if changepoint <= 3 or (changepoint >= len(t) - 4):
    #     return np.array([changepoint_idx])
    # lhs = find_all_poly_splits(t[:changepoint+1], val[:changepoint+1], start, changepoint_idx)
    # rhs = find_all_poly_splits(t[changepoint:], val[changepoint:], changepoint_idx, end) 
    # lhs = lhs if lhs is not None else np.zeros((0,))
    # rhs = rhs if rhs is not None else np.zeros((0,))
    # return np.concatenate([lhs, np.array([changepoint_idx]), rhs])

def find_poly_changepoints_jointly(t, val_list, tol=1e-16, chg_pts_cand_list=None):
    assert chg_pts_cand_list is not None, 'Unsupported initiazliation'
    chg_pts_cand = np.sort(np.unique(np.concatenate(chg_pts_cand_list)))
    if len(chg_pts_cand) == 2:
        return chg_pts_cand
    test_ind = 1
    while chg_pts_cand[test_ind] != len(t) - 1:
        if chg_pts_cand[test_ind] - chg_pts_cand[test_ind - 1] <= 4:
            chg_pts_cand = np.delete(chg_pts_cand, test_ind)
            continue
        test_start = int(chg_pts_cand[test_ind-1])
        test_end = int(chg_pts_cand[test_ind+1]+1) 
        errs = np.polyfit(t[test_start:test_end], val_list.T[test_start:test_end], deg=3, full=True)[1]
        if np.sum(errs) < tol:
            chg_pts_cand = np.delete(chg_pts_cand, test_ind)
        else:
            test_ind += 1
    if chg_pts_cand[test_ind] - chg_pts_cand[test_ind - 1] <= 4:
        chg_pts_cand = np.delete(chg_pts_cand, test_ind - 1)
    return chg_pts_cand

# all indices in chg_pts are INCLUSIVE
def fit_for_chg_pts(t, val_list, chg_pts):
    def get_err(idx1, idx2):
        start = int(chg_pts[idx1])
        end = int(chg_pts[idx2] + 1)
        err = np.polyfit(t[start:end], val_list.T[start:end], deg=3, full=True)[1].sum()
        return err
    err = get_err(0, 1)
    if len(chg_pts) == 2:
        return err
    all_errs = [err]
    for i in range(1, len(chg_pts) - 1):
        # Technically shouldn't include the first error, double counting and such...but oh well
        new_err = get_err(i, i+1)
        all_errs.append(new_err)
    err = np.sum(all_errs)
    return err

# Larger reg_mul -> fewer chg points
def get_opt_number_chg_pts_jointly(t, val_list, min_timesteps=5, test_tol_list=TEST_TOLS, reg_mul=1):
    if val_list.shape[-1] < min_timesteps:
        return np.array([0, val_list.shape[-1] - 1])
    
    #base_errs = np.polyfit(t, val_list.T, deg=3, full=True)[1].sum()
    #base_errs = np.linalg.norm(val_list - val_list[:, 0][:, np.newaxis])/val_list.size
    # breakpoint()
    # When error is higher -> fewer chg points
    #test_tol_list = [base_errs * reg_mul]


    chg_pts_cand_list = []
    for val in val_list:
        splits = find_all_poly_splits(t, val)
        if splits is None:
            splits = np.zeros((0,))
        else:
            splits = np.sort(splits)
        chg_pts_cand = np.concatenate([np.array([0]), splits, np.array([len(t) - 1])])
        chg_pts_cand_list = [chg_pts_cand]
        break
    chg_pts_cand = np.sort(np.unique(np.concatenate(chg_pts_cand_list)))
    if len(chg_pts_cand) == 2:
        return chg_pts_cand
    assert len(chg_pts_cand) > 2, 'Invalid number of chg pts admitted'

    opt_chg_pts = []
    test_error = np.inf
    err_list = []
    for tol in test_tol_list:
        tol_chg_pts = find_poly_changepoints_jointly(t, val_list, tol, chg_pts_cand_list)
        # Regularization
        err = len(tol_chg_pts)*reg_mul
        fit_errs = fit_for_chg_pts(t, val_list, tol_chg_pts)
        err += fit_errs
        err_list.append(err)
        if err < test_error:
            test_error = err
            opt_chg_pts = tol_chg_pts
    assert len(opt_chg_pts) >= 2, 'Invalid number of change points'
    return opt_chg_pts#, err_list

def assign_primitives(input, interp_vals, hist_only=False, extrap=False, min_timesteps=5):
    if hist_only:
        interp_vals = interp_vals[:, :11]
    
    def change_point_ids(change_points):
        change_points = np.array([[start, end] for start, end in zip(change_points[:-1], change_points[1:])])
        # Make last point exclusive
        change_points[-1, -1] += 1
        state_vals = [i for i in range(len(change_points))]
        return np.concatenate([np.array(change_points), np.array(state_vals)[:, np.newaxis]], axis=-1).astype(np.float32)

    
    def compute_single(y_single, agent_idxs):
        assert not np.isnan(y_single).any(), 'No nan allowed here'
        t = np.arange(len(y_single))
        if hist_only:
            change_points = np.array([0.0, 10])
        elif extrap:
            change_points = np.array([0.0, 11, 90])
        else:
            change_points = get_opt_number_chg_pts_jointly(t, y_single[:, :3].T, min_timesteps=min_timesteps)
        single_idx = np.array(agent_idxs)
        single = change_point_ids(change_points)
        return single, single_idx

    single_iter = get_singles(interp_vals, interp_only=(not extrap))

    #if args.parallel:
    if False:
        from joblib import Parallel, delayed    
        with tqdm_joblib(tqdm(desc='Assigning singles', leave=False, total=len(single_iter[0]), disable=True)) as progress_bar:
            single_outs = Parallel(n_jobs=args.nproc, batch_size=25)(delayed(compute_single)(y_single, agent_idxs)
                for y_single, agent_idxs in zip(*single_iter))
    else:
        single_outs = []
        for y_single, agent_idxs in tqdm(zip(*single_iter), 'Assigning singles', leave=False, total=len(single_iter[0]), disable=True):
            out = compute_single(y_single, agent_idxs)
            single_outs.append(out)
        
    singles = [x[0] for x in single_outs]
    single_idxs = [x[1] for x in single_outs]
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

    def compute_pair(y_pair, agent_idxs):
        assert not np.isnan(y_pair).any(), 'No nan allowed here'
        t = np.arange(len(y_pair))
        if hist_only:
            change_points = np.array([0.0, 10])
        elif extrap:
            change_points = np.array([0.0, 11, 90])
        else:
            change_points = get_opt_number_chg_pts_jointly(t, y_pair[:, :6].T, min_timesteps=min_timesteps)
        pair_idx = np.array(agent_idxs)
        pair = change_point_ids(change_points)
        return pair, pair_idx

    pair_iter = get_encounters(interp_vals, interp_only=(not extrap))

    # if args.parallel:
    if False:
        from joblib import Parallel, delayed    
        with tqdm_joblib(tqdm(desc='Assigning pairs', leave=False, total=len(pair_iter[0]), disable=True)) as progress_bar:
            pair_outs = Parallel(n_jobs=args.nproc, batch_size=500)(delayed(compute_pair)(y_pair, agent_idxs)
                for y_pair, agent_idxs in zip(*pair_iter))
    else:
        pair_outs = []
        for y_pair, agent_idxs in tqdm(zip(*pair_iter), 'Assigning pairs', leave=False, total=len(pair_iter[0]), disable=True):
            out = compute_pair(y_pair, agent_idxs)
            pair_outs.append(out)
        
    pairs = [x[0] for x in pair_outs]
    pair_idxs = [x[1] for x in pair_outs]
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
    hist_suffix = '_hist' if hist_only else '_extrap' if extrap else ''

    # Now, we actually output the labeled primitives
    for split in splits:
        CACHE_SUBDIR = os.path.join(CACHE_DIR, split, 'primitives_spline')
        os.makedirs(CACHE_SUBDIR, exist_ok=True)
        for shard_idx in range(10):
            metas, inputs, interp_vals = load_base(split, base_path, cache_path, shards, num_scenarios, hist_only or extrap, shard_idx=shard_idx)
            msg = f'Processing scenarios for {split}, shard {shard_idx}'
            start_time = time.time()

            if args.parallel:
                from joblib import Parallel, delayed    
                with tqdm_joblib(tqdm(desc=msg, leave=True, total=len(metas))) as progress_bar:
                    all_outs = Parallel(n_jobs=args.nproc, batch_size=1)(delayed(assign_primitives)(
                        input, interp_val, hist_only=hist_only, extrap=extrap, min_timesteps=args.min_timesteps)
                        for _, input, interp_val in zip(metas, inputs, interp_vals))
            else:
                all_outs = []
                for _, input, interp_val in tqdm(zip(metas, inputs, interp_vals), msg, total=len(metas)):
                    out = assign_primitives(input, interp_val, hist_only=hist_only, extrap=extrap, min_timesteps=args.min_timesteps)
                    all_outs.append(out)
            
            print(f'Done in {time.time() - start_time}')

            # n_single = [(x['singles'][:, -1] == 0).sum() for x in all_outs]
            # n_pairs = [(x['pairs'][:, -1] == 0).sum() for x in all_outs]
            # n_single_prims = [len(x['singles']) for x in all_outs]
            # n_pair_prims = [len(x['pairs']) for x in all_outs]
            # single_cnts_ = [np.unique(np.unique(x['singles'][:, :1], axis=0, return_counts=True)[1], return_counts=True) for x in all_outs]
            # pair_cnts_ = [np.unique(np.unique(x['pairs'][:, :2], axis=0, return_counts=True)[1], return_counts=True) for x in all_outs]
            # single_cnts = {}
            # pair_cnts = {}
            # for all_prims, all_cnts in single_cnts_:
            #     for n_prim, cnt in zip(all_prims, all_cnts):
            #         if n_prim in single_cnts:
            #             single_cnts[n_prim] += cnt
            #         else:
            #             single_cnts[n_prim] = cnt
            # for all_prims, all_cnts in pair_cnts_:
            #     for n_prim, cnt in zip(all_prims, all_cnts):
            #         if n_prim in pair_cnts:
            #             pair_cnts[n_prim] += cnt
            #         else:
            #             pair_cnts[n_prim] = cnt
            
            # print(sum(n_single))
            # print(sum(n_single_prims))
            # print(sum(n_pairs))
            # print(sum(n_pair_prims))
            # print(single_cnts)
            # print(pair_cnts)
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
    parser.add_argument('--extrap', action='store_true')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--nproc', type=int, default=20)
    parser.add_argument('--min_timesteps', type=int, default=5)
    parser.add_argument('--recache', action='store_true', help='Ignore/rebuild cache')
    args = parser.parse_args()

    assert not (args.extrap and args.hist_only), 'Only one of extrap and hist_only permitted'

    extract_primitives(args, args.base_path, args.cache_path, args.num_scenarios,
                           args.shards, args.num_shards, args.hist_only, args.extrap)
