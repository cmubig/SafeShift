import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import time
import pandas as pd

from matplotlib import cm
from operator import itemgetter
from tqdm import tqdm

from utils.common import load_infos, compute_velocities, VISUAL_OUT_DIR, CACHE_DIR, IPOS_XY_IDX
from aggregate_features import agg_features

def process_file(
    scenario_path: str, scenario_meta: str, interp_trajectories: np.array,
    other_interp_trajectories: np.array,
    conflict_points: list, cluster_anomaly: dict, other_cluster_anomaly: dict, load_type: str, plot: bool = False, 
    tag: str ='temp', hist_len: int = 11, compute_individual_features: bool = False, 
    compute_interaction_features: bool = False, compute_frenetp_features: bool = True, 
):
    # Load the scenario 
    with open(scenario_path, 'rb') as f:
        scenario = pkl.load(f)
    
    assert (other_interp_trajectories is None) or load_type == 'asym', 'Must load asym in this case'
    
    # TODO: start by checking if hist only, to enforce only inspecting first 11 timesteps
    if load_type == 'ho':
        interp_trajectories = interp_trajectories[:, :hist_len, :]
        # conflict_points is just scene: static=stop signs, crosswalk; lane = intersection, dynamic=stop points

    # In asym case, interp_velocities represents gt and other_interp_velocities represents fe
    interp_velocities = compute_velocities(interp_trajectories)
    other_interp_velocities = compute_velocities(other_interp_trajectories)

    # Individual State Measures
    individual_features = {}
    if compute_individual_features:
        from utils.individual_measures import compute_individual_features
        traj_in = interp_trajectories if load_type != 'asym' else other_interp_trajectories
        vel_in = interp_velocities if load_type != 'asym' else other_interp_velocities
        anomaly_in = cluster_anomaly if load_type != 'asym' else other_cluster_anomaly
        individual_features = compute_individual_features(
            scenario=scenario, conflict_points=conflict_points, load_type=load_type, 
            interp_trajectories=traj_in, interp_velocities=vel_in, 
            cluster_anomaly=anomaly_in,
            plot=plot, tag=tag, timesteps=interp_trajectories.shape[1], hz=10)

    # Agent-to-Agent State
    interaction_features = {}
    # TODO: see if we need to incorporate the assymetric cluster anomaly? For now, stick to GT I think
    if compute_interaction_features:
        from utils.interaction_measures import compute_interaction_features, Status
        interaction_features = compute_interaction_features(
            scenario=scenario, conflict_points=conflict_points,
            load_type=load_type, interp_trajectories=interp_trajectories, 
            interp_velocities=interp_velocities, 
            other_interp_trajectories=other_interp_trajectories,
            other_interp_velocities=other_interp_velocities,
            plot=plot, tag=tag,
            timesteps=interp_trajectories.shape[1], hz=10,
            cluster_anomaly=cluster_anomaly)
    
    # Frenet+ features
    frenetp_features = {}
    if compute_frenetp_features:
        from tools.scenario_identification.utils.frenetp_measures import compute_frenetp_features
        frenetp_features = compute_frenetp_features(
            scenario, interp_trajectories, interp_velocities, load_type, hist_len)

    return {
        "individual_features": individual_features, "interaction_features": interaction_features,
        "frenetp_features": frenetp_features,
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='/av_shared_ssd/mtr_process_ssd')
    parser.add_argument('--split', type=str, default='training', choices=['training', 'validation', 'testing'])
    parser.add_argument('--num_scenarios', type=int, default=-1)
    parser.add_argument('--nproc', type=int, default=10)
    parser.add_argument('--shards', type=int, default=10)
    parser.add_argument('--shard_idx', type=int, nargs='+',required=True)
    
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--plot', action='store_true')
    
    parser.add_argument('--load_type', required=True, choices=['gt', 'fe', 'ho', 'asym'])

    parser.add_argument('--compute_frenetp_features', action='store_true')
    parser.add_argument('--compute_individual_features', action='store_true')
    parser.add_argument('--compute_interaction_features', action='store_true')

    args = parser.parse_args()
    args.use_interp = True

    VISUAL_OUT_SUBDIR = os.path.join(VISUAL_OUT_DIR, args.split, f"{__file__.split('.')[0]}")
    os.makedirs(VISUAL_OUT_SUBDIR, exist_ok=True)
    
    load_lane_cache, load_conflict_points, load_cluster_anomaly = False, False, False
    if args.compute_individual_features:
        load_conflict_points = True
        load_cluster_anomaly = True

    if args.compute_interaction_features:
        #load_lane_cache, load_conflict_points = True, True
        load_conflict_points = True
        load_cluster_anomaly = True

    cache_path = os.path.join(CACHE_DIR, args.split)
    if args.load_type == 'asym':
        infos_gt = load_infos(base_path=args.base_path, cache_path=cache_path, split=args.split, 
            shard_idx=args.shard_idx, shards=args.shards, num_scenarios=args.num_scenarios, 
            load_lane_cache=load_lane_cache, load_conflict_points=load_conflict_points,
            load_type='gt', load_cluster_anomaly=load_cluster_anomaly
        )
        infos_fe = load_infos(base_path=args.base_path, cache_path=cache_path, split=args.split, 
            shard_idx=args.shard_idx, shards=args.shards, num_scenarios=args.num_scenarios, 
            load_lane_cache=load_lane_cache, load_conflict_points=load_conflict_points,
            load_type='fe', load_cluster_anomaly=load_cluster_anomaly
        )

        total_n = len(infos_gt['inputs'])
        zipped = zip(
            infos_gt['inputs'], infos_gt['metas'], infos_gt['interp_trajectories'], infos_fe['interp_trajectories'],
            infos_gt['conflict_points'], infos_gt['cluster_anomaly'], infos_fe['cluster_anomaly']
        )
    else:
        infos = load_infos(base_path=args.base_path, cache_path=cache_path, split=args.split, 
            shard_idx=args.shard_idx, shards=args.shards, num_scenarios=args.num_scenarios, 
            load_lane_cache=load_lane_cache, load_conflict_points=load_conflict_points,
            load_type=args.load_type, load_cluster_anomaly=load_cluster_anomaly
        )

        total_n = len(infos['inputs'])
        zipped = zip(
            infos['inputs'], infos['metas'], infos['interp_trajectories'], [None]*total_n,
            infos['conflict_points'], infos['cluster_anomaly'], [None]*total_n
        )

    msg = f'Processing {args.split} split scenarios...'
    start = time.time()
    if args.parallel:
        from joblib import Parallel, delayed    
        all_outs = Parallel(n_jobs=args.nproc, batch_size=4)(delayed(process_file)(
            scenario_path=path, 
            scenario_meta=meta, 
            interp_trajectories=interp_trajs, 
            other_interp_trajectories=other_interp_trajs,
            conflict_points=conflict_points, 
            cluster_anomaly=cluster_anomaly,
            other_cluster_anomaly=other_cluster_anomaly,
            load_type=args.load_type,
            plot=args.plot, 
            tag=f"{VISUAL_OUT_SUBDIR}/{s.split('.')[0]}", 
            compute_frenetp_features=args.compute_frenetp_features,
            compute_individual_features=args.compute_individual_features,
            compute_interaction_features=args.compute_interaction_features)
            for (s, path), meta, interp_trajs, other_interp_trajs, conflict_points, cluster_anomaly, other_cluster_anomaly \
                in tqdm(zipped, desc=msg, total=total_n))
    else:
        all_outs = []
        for (s, path), meta, interp_trajs, other_interp_trajs, conflict_points, cluster_anomaly, other_cluster_anomaly \
              in tqdm(zipped, msg, total=total_n):
            out = process_file(
                scenario_path=path, 
                scenario_meta=meta, 
                interp_trajectories=interp_trajs,
                other_interp_trajectories=other_interp_trajs,
                conflict_points=conflict_points,
                cluster_anomaly=cluster_anomaly,
                other_cluster_anomaly=other_cluster_anomaly,
                load_type=args.load_type,
                plot=args.plot, 
                tag=f"{VISUAL_OUT_SUBDIR}/{s.split('.')[0]}", 
                compute_frenetp_features=args.compute_frenetp_features,
                compute_individual_features=args.compute_individual_features,
                compute_interaction_features=args.compute_interaction_features)
            all_outs.append(out)
    print(f"\tProcessing took {time.time() - start} seconds.")

    # Unpack and save 
    if args.cache:
        CACHE_SUBDIR = os.path.join(cache_path, 'features')
        os.makedirs(CACHE_SUBDIR, exist_ok=True)
        suffix = (f'_{args.load_type}_shard{args.shard_idx[0]}_{args.shards}.npz')
        
        if args.compute_individual_features:
            cache_filepath = os.path.join(CACHE_SUBDIR, f'individual_agg{suffix}')
            individual_features = [state['individual_features'] for state in all_outs]
            agg_individual, _ = agg_features(individual_features=individual_features)
            with open(cache_filepath, 'wb') as f:
                np.savez_compressed(f, agg_individual) 

        if args.compute_interaction_features:
            cache_filepath = os.path.join(CACHE_SUBDIR, f'interaction_agg{suffix}')
            interaction_features = [state['interaction_features'] for state in all_outs]
            _, agg_interaction = agg_features(interaction_features=interaction_features)
            with open(cache_filepath, 'wb') as f:
                np.savez_compressed(f, agg_interaction)
               
        if args.compute_frenetp_features:
            cache_filepath = os.path.join(CACHE_SUBDIR, f'frenetp_features{suffix}')
            frenetp_features = [state['frenetp_features'] for state in all_outs]
            with open(cache_filepath, 'wb') as f:
                np.savez_compressed(f, frenetp_features)

        # TODO: compute and save context features
        print("Done.")