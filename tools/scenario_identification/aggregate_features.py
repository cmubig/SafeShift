import numpy as np
import os
import pickle as pkl
import time
import pandas as pd
import json

from tqdm import tqdm

from utils.common import compute_velocities, VISUAL_OUT_DIR, CACHE_DIR, IPOS_XY_IDX

# from utils.individual_measures import normalize_individual_features

def get_mean_std(features, feature_name, feature_subname = None):
    unpacked_feature = []
    if not feature_subname is None:
        for i in range(len(features)):
            unpacked_feature += np.concatenate(features[i][feature_name][feature_subname]).tolist()
    else:
        for i in range(len(features)):
            unpacked_feature += np.concatenate(features[i][feature_name]).tolist()
    unpacked_feature = np.asarray([unpacked_feature]).reshape(-1)
    unpacked_feature = unpacked_feature[np.isfinite(unpacked_feature)]
    return unpacked_feature.mean(), unpacked_feature.std()

def agg_features(individual_features=None, interaction_features=None):

    if individual_features is not None:
        # Around 400 it/sec
        individual_agg = []
        for agent_features in tqdm(individual_features, 'Processing individual', total=len(individual_features)):
            agg_out = {}
            # Identifying features first
            agg_out['agent_types'] = agent_features['agent_types']
            agg_out['agent_ids'] = agent_features['valid_agents']
            agg_out['speed_val'] = [x.max() for x in agent_features['speed']['values']]
            agg_out['speed_lane_limit_diff_val'] = [x[mask].max() if mask.any() else 0 for x, mask in zip(agent_features['speed']['lane_limit_diff'], agent_features['in_lane'])]
            # These values aren't very good to be honest, showing up as like 6 gs and such
            agg_out['acc_x_val'] = [np.abs(x).max() for x in agent_features['acc']['ax']]
            agg_out['acc_y_val'] = [np.abs(x).max() for x in agent_features['acc']['ay']]
            agg_out['jerk_x_val'] = [np.abs(x).max() for x in agent_features['jerk']['jx']]
            agg_out['jerk_y_val'] = [np.abs(x).max() for x in agent_features['jerk']['jy']]
            # Sometimes happens even when conflict point is None, so maybe just add it?
            #agg_out['wp_val'] = [x[np.isfinite(x)][dist[np.isfinite(x)] < agent_scene_conflict_dist_thresh].sum() for x, dist in zip(agent_features['waiting_period']['intervals'], agent_features['waiting_period']['min_dists_conflict_points'])]
            agg_out['wp_val'] = [x[np.isfinite(x)].sum() for x in agent_features['waiting_period']['intervals']]
            agg_out['in_lane_val'] = [x.mean() for x in agent_features['in_lane']]
            agg_out['traj_anomaly_val'] = agent_features['traj_anomaly']
            for k in agg_out.keys():
                agg_out[k] = np.array(agg_out[k])
            individual_agg.append(agg_out)
    else:
        individual_agg = None
    
    if interaction_features is not None:
        interaction_agg = []
        # Around 30 it/sec
        for pair_features in tqdm(interaction_features, 'Processing interactions', total=len(interaction_features)):
            agg_out = {}
            # Identifying features first
            agg_out['pair_status'] = [x.name for x in pair_features['status']]
            agg_out['pair_ids'] = pair_features['agent_ids']
            agg_out['pair_types'] = pair_features['agent_types']
            agg_out['thw_val'] = [1/np.min(x) if np.array(x).size > 0 else 0 for x in pair_features['thw']]
            agg_out['ttc_val'] = [1/np.min(x) if np.array(x).size > 0 else 0 for x in pair_features['ttc']]
            agg_out['drac_val'] = [np.max(x) if np.array(x).size > 0 else 0 for x in pair_features['drac']]
            agg_out['scene_mttcp_val'] = [1/np.min(x) if np.array(x).size > 0 else 0 for x in pair_features['scene_mttcp']]
            agg_out['agent_mttcp_val'] = [1/np.min(x) if np.array(x).size > 0 else 0 for x in pair_features['agent_mttcp']]
            agg_out['collisions_val'] = [np.sum(x) if np.array(x).size > 0 else 0 for x in pair_features['collisions']]
            agg_out['traj_pair_anomaly_val'] = pair_features['traj_pair_anomaly']
            for k in agg_out.keys():
                agg_out[k] = np.array(agg_out[k])
            interaction_agg.append(agg_out)
    else:
        interaction_agg = None
    return individual_agg, interaction_agg

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

    parser.add_argument('--load_type', required=True, choices=['gt', 'fe', 'ho'])

    parser.add_argument('--compute_individual_scores', action='store_true')
    parser.add_argument('--compute_interaction_scores', action='store_true')

    args = parser.parse_args()
    raise NotImplementedError('Standalone aggregate features is incomplete')

    VISUAL_OUT_SUBDIR = os.path.join(VISUAL_OUT_DIR, args.split, f"{__file__.split('.')[0]}")
    os.makedirs(VISUAL_OUT_SUBDIR, exist_ok=True)

    cache_path = os.path.join(CACHE_DIR, args.split, 'features')
    suffix = (f'_{args.load_type}_shard{args.shard_idx[0]}_{args.shards}.npz')
    
    # Loading test shard 0 takes ~2 minutes, not *that* atrocious
    individual_features = None
    if args.compute_individual_scores:
        start = time.time()
        print(f"Loading individual features...")
        individual_features_filepath = os.path.join(cache_path, f'individual_features{suffix}')
        individual_features = np.load(individual_features_filepath, allow_pickle=True)['arr_0']
        print(f"\t...process took {time.time() - start}")
    
    interaction_features = None
    if args.compute_interaction_scores:
        start = time.time()
        print(f"Loading interaction features...")
        interaction_features_filepath = os.path.join(cache_path, f'interaction_features{suffix}')
        interaction_features = np.load(interaction_features_filepath, allow_pickle=True)['arr_0']
        print(f"\t...process took {time.time() - start}")

    msg = f'Processing {args.split} split scenarios...'
    start = time.time()
    agg_individual, agg_interaction = agg_features(individual_features, interaction_features)
    print(f"\tProcessing took {time.time() - start} seconds.")

    # TODO: see how hefty these are to save, and whether it makes sense to un-shard things
    # They're tiny! Can totally un-shard them :)

    # Unpack and save 
    if args.cache:
        CACHE_SUBDIR = os.path.join(cache_path, 'features_agg')
        os.makedirs(CACHE_SUBDIR, exist_ok=True)
        suffix = (f'_{args.load_type}_shard{args.shard_idx[0]}_{args.shards}.npz')
        
        if args.compute_individual_scores:
            cache_filepath = os.path.join(CACHE_SUBDIR, f'individual_agg{suffix}')
            with open(cache_filepath, 'wb') as f:
                np.savez_compressed(f, agg_individual) 

        if args.compute_interaction_scores:
            cache_filepath = os.path.join(CACHE_SUBDIR, f'interaction_agg{suffix}')
            with open(cache_filepath, 'wb') as f:
                np.savez_compressed(f, agg_interaction)
    print("Done.")