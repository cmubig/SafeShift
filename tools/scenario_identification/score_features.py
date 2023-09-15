import numpy as np
import os
import pickle as pkl
import time
import pandas as pd
import json
from matplotlib import pyplot as plt
import seaborn as sns

from tqdm import tqdm

from utils.common import load_infos, compute_velocities, VISUAL_OUT_DIR, CACHE_DIR, IPOS_XY_IDX
from utils.visualization import plot_dynamic_map_infos, plot_static_map_infos

def plot_scene(scenario, 
               scores_gt, total_score_gt, interp_gt, 
               scores_fe, total_score_fe, interp_fe,
               total_score_asym_combined,
               tag):
        num_windows = 2
        point_size = 1
        alpha = 0.5
        fig, axs = plt.subplots(1, num_windows, figsize=(5 * num_windows, 5 * 1))

        static_map_infos = scenario['map_infos']
        dynamic_map_infos = scenario['dynamic_map_infos']
        to_predict = scenario['tracks_to_predict']['track_index']

        plot_static_map_infos(static_map_infos, axs, num_windows=num_windows)
        plot_dynamic_map_infos(dynamic_map_infos, axs, num_windows=num_windows)
        
        for ax, interp_trajectories, scores, total_score, load_type in zip(axs.reshape(-1), [interp_gt, interp_fe], [scores_gt, scores_fe], 
                                                  [total_score_gt, total_score_fe], ['gt', 'fe']):
            score_vals = np.array([x for x in scores.values()])
            if not np.allclose(score_vals, 0.0):
                alphas = (score_vals - score_vals.min()) / (score_vals.max() - score_vals.min())
            alphas = np.clip(alphas, 0.2, 1.0)
            alphas = {k: v for k, v in zip(scores.keys(), alphas)}
            for n in scores.keys():
                # Uses interpolated and extrapolated regions 
                if np.any(np.isnan(interp_trajectories[n, :, IPOS_XY_IDX])):
                    continue

                pos = interp_trajectories[n, :, IPOS_XY_IDX].T
                if load_type == 'gt':
                    # Uses interpolated regions only
                    mask = np.where(interp_trajectories[n, :, -1] == 1)[0]
                    if mask.shape[0] == 0:
                        continue
                    pos = interp_trajectories[n, mask[0]:(mask[-1]+1), IPOS_XY_IDX].T

                color = 'blue' if n not in to_predict else 'green'
                ax.plot(pos[:, 0], pos[:, 1], color=color, alpha=alphas[n])

            ax.set_title(f'Scene Score {load_type.upper()}: {round(total_score, 3)}')
            ax.set_xticks([])
            ax.set_yticks([])
            
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(
            f"{tag}_score_{round(total_score_asym_combined, 3)}.png", 
            dpi=300, bbox_inches='tight')
        plt.clf()


def compute_score(individual_gt, interaction_gt, gt_infos,
                 individual_fe, interaction_fe, fe_infos, 
                 individual_asym, interaction_asym, plot=False):

    def simple_individual_score(feat, k=1):
        accs = 0.1*np.linalg.norm([feat['acc_x_val'], feat['acc_y_val']])
        jerk = 0.01*np.linalg.norm([feat['jerk_x_val'], feat['jerk_y_val']])
        return min(10, max(0, feat['speed_lane_limit_diff_val'])) + \
               min(10, accs) + \
               min(10, jerk) + \
               min(10, 0.1*feat['speed_val']) + \
               min(5, feat['traj_anomaly_val']) + \
               min(7.75, np.sqrt(feat['wp_val'])) + \
               1 - feat['in_lane_val']
    
    valid_agent_ids = [set(x['agent_ids']).intersection(set(y['agent_ids'])) for x, y in zip(individual_gt, individual_fe)]
    scene_scores = []
    for scene_x, scene_y, scene_z, ids in tqdm(zip(individual_gt, individual_fe, individual_asym, valid_agent_ids), 
                                               'Processing individual', total=len(individual_gt)):
        traj_scores = {'gt': [], 'fe': [], 'asym': [], 'combined': [], 'asym_combined': [], 'fe_lim': [], 'id': []} 
        for agent_id in ids:
            x_idx = np.where(scene_x['agent_ids'] == agent_id)[0].item()
            y_idx = np.where(scene_y['agent_ids'] == agent_id)[0].item()
            z_idx = np.where(scene_z['agent_ids'] == agent_id)[0].item()
            feat_x = {k: scene_x[k][x_idx] for k in scene_x.keys()}
            feat_y = {k: scene_y[k][y_idx] for k in scene_y.keys()}
            feat_z = {k: scene_z[k][z_idx] for k in scene_z.keys()}

            score_x = simple_individual_score(feat_x)
            score_y = simple_individual_score(feat_y)
            score_z = simple_individual_score(feat_z)
            traj_scores['gt'].append(score_x)
            traj_scores['fe'].append(score_y)
            traj_scores['asym'].append(score_z)
            traj_scores['asym_combined'].append(max(score_x, score_z))

            # For limited ones, keep individual state the same as gt
            traj_scores['fe_lim'].append(score_x)
            # For now, just do max of either
            traj_scores['combined'].append(max(score_x, score_y))
            traj_scores['id'].append(agent_id)
        for k in traj_scores.keys():
            traj_scores[k] = np.array(traj_scores[k])
        scene_scores.append(traj_scores)
    
    # Using implementation from Table III of Automated Analysis
    def simple_interaction_score(feat, k=1):
        return min(2, feat['thw_val']) + \
               min(4, feat['scene_mttcp_val']) + \
               min(4, feat['agent_mttcp_val']) + \
               min(2*k, feat['ttc_val']) + \
               min(2*k, k/5 * feat['drac_val']) + \
               k*feat['collisions_val'] + \
               min(5, 0.1*feat['traj_pair_anomaly_val'])

    # gt_greater = 0
    # fe_greater = 0
    final_scores = {
        'gt': [],
        'fe': [],
        'asym': [],
        'combined': [],
        'asym_combined': [],
        'fe_lim': [],

        'fe_trajs': [],
        'asym_combined_trajs': [],
        'gt_trajs': [],
        'fe_traj_weights': [],
        'asym_combined_traj_weights': [],
        'gt_traj_weights': [],
        'fe_idxs': [],
        'asym_combined_idxs': [],
        'gt_idxs': []
    }
    for scene_x, scene_y, scene_z, meta_x, interp_x, interp_y, traj_scores, ids, (_, scenario_path) in \
            tqdm(zip(interaction_gt, interaction_fe, interaction_asym, gt_infos['metas'],
                     gt_infos['interp_trajectories'], fe_infos['interp_trajectories'],
                     scene_scores, valid_agent_ids, gt_infos['inputs']), 'Processing interaction', total=len(interaction_gt)):
        with open(scenario_path, 'rb') as f:
            scenario = pkl.load(f)
        assert (scene_x['pair_ids'] == scene_y['pair_ids']).all()
        # First element points to the idx in scene_z['pair_ids'] that corresponds to i FE, j GT; second is opposite
        z_idx_map = {}
        for pair_idx, pair_id in enumerate(scene_z['pair_ids']):
            if pair_id[0] < pair_id[1]:
                key = (pair_id[0], pair_id[1])
                inner_idx = 0
            else:
                key = (pair_id[1], pair_id[0])
                inner_idx = 1
            if key not in z_idx_map:
                z_idx_map[key] = [0, 0]
            z_idx_map[key][inner_idx] = pair_idx

        # Use gt score for individual base, since constant velocity extrapolation won't add sus dynamics
        tot_scores_gt = {ids_: traj_score for ids_, traj_score in zip(ids, traj_scores['gt'])}
        tot_scores_fe = {ids_: traj_score for ids_, traj_score in zip(ids, traj_scores['fe'])}
        tot_scores_asym = {ids_: traj_score for ids_, traj_score in zip(ids, traj_scores['asym'])}
        tot_scores_combined = {ids_: traj_score for ids_, traj_score in zip(ids, traj_scores['combined'])}
        tot_scores_asym_combined = {ids_: traj_score for ids_, traj_score in zip(ids, traj_scores['asym_combined'])}
        tot_scores_fe_lim = {ids_: traj_score for ids_, traj_score in zip(ids, traj_scores['fe_lim'])}
        for idx in range(len(scene_x['pair_ids'])):
            i, j = scene_x['pair_ids'][idx]
            # Skip ones that are not valid
            if i not in ids or j not in ids:
                continue
            assert i < j, 'Confusingly, i >= j somehow'

            feat_x = {k: scene_x[k][idx] for k in scene_x.keys()}
            feat_y = {k: scene_y[k][idx] for k in scene_y.keys()}
            
            z_key = (i, j)
            z_idx0, z_idx1 = z_idx_map[z_key]
            feat_z0 = {k: scene_z[k][z_idx0] for k in scene_z.keys()}
            feat_z1 = {k: scene_z[k][z_idx1] for k in scene_z.keys()}
            status_x, status_y = feat_x['pair_status'], feat_y['pair_status']
            score_x = simple_interaction_score(feat_x)
            score_y = simple_interaction_score(feat_y)
            score_z0 = simple_interaction_score(feat_z0)
            score_z1 = simple_interaction_score(feat_z1)
            # Symmetric scoring for interactions, simplifying assumption
            tot_scores_gt[i] += score_x
            tot_scores_gt[j] += score_x
            tot_scores_fe[i] += score_y
            tot_scores_fe[j] += score_y
            tot_scores_combined[i] += max(score_x, score_y)
            tot_scores_combined[j] += max(score_x, score_y)

            # Assymetric scoring
            tot_scores_asym[i] += score_z0
            tot_scores_asym[j] += score_z1

            tot_scores_asym_combined[i] += max(score_x, score_z0)
            tot_scores_asym_combined[j] += max(score_x, score_z1)
            
            feat_fe_lim = {k: max(feat_x[k], feat_y[k]) for k in feat_x.keys() if '_val' in k}
            score_fe_lim = simple_interaction_score(feat_fe_lim)
            tot_scores_fe_lim[i] += score_fe_lim
            tot_scores_fe_lim[j] += score_fe_lim

        # Traj aggregation things:
        # 1. Just take max
        # 2. Take mean or max over to_predict
        # 3. Take distance-weighted mean to to_predict

        to_predict = meta_x['tracks_to_predict']['track_index'] 
        def weight_score(score, use_gt=False, use_fe=False):
            assert use_gt or use_fe, 'One must be true at least'
            traj_gt = interp_x
            to_predict_gt = traj_gt[to_predict]
            traj_fe = interp_y
            to_predict_fe = traj_fe[to_predict]
            weights = {}
            for agent_idx in score.keys():
                if agent_idx in to_predict:
                    weights[agent_idx] = 1
                    continue
                agent_gt = traj_gt[agent_idx]
                agent_fe = traj_fe[agent_idx]
                min_dists = []
                for comp_gt, comp_fe in zip(to_predict_gt, to_predict_fe):
                    mask_gt = agent_gt[:, -1].astype(bool) & comp_gt[:, -1].astype(bool)
                    min_dist_gt = np.linalg.norm(comp_gt[mask_gt, :2] - agent_gt[mask_gt, :2], axis=-1)
                    min_dist_gt = np.inf if not min_dist_gt.size > 0 else min_dist_gt.min()
                    min_dist_fe = np.linalg.norm(comp_fe[:, :2] - agent_fe[:, :2], axis=-1)
                    min_dist_fe = min_dist_fe.min()
                    min_dist = min(min_dist_fe, min_dist_gt) if use_gt and use_fe else min_dist_fe if use_fe else min_dist_gt
                    min_dists.append(min_dist)
                min_dist = np.min(min_dists)
                weights[agent_idx] = 1/(1 + min_dist)
            return weights
        
        weights_gt = weight_score(tot_scores_gt, use_gt=True)
        weights_fe = weight_score(tot_scores_fe, use_fe=True)
        # For asym, just use both FE and GT for now, for the weights
        weights_asym = weight_score(tot_scores_asym, use_fe=True, use_gt=True)
        weights_combined = weight_score(tot_scores_combined, use_fe=True, use_gt=True)
        weights_asym_combined = weight_score(tot_scores_asym_combined, use_fe=True, use_gt=True)
        weights_fe_lim = weight_score(tot_scores_fe_lim, use_fe=True, use_gt=True)
        # max_score_gt = np.array([x for x in tot_scores_gt.values()]).max()
        # TODO: what to divide by?
        # denom = len(to_predict)
        # denom = len(tot_scores_gt)
        denom = len(to_predict) + np.sqrt(len(tot_scores_gt) - len(to_predict))

        weighted_gt = np.sum([weights_gt[k]*tot_scores_gt[k] for k in tot_scores_gt.keys()])/denom
        weighted_fe = np.sum([weights_fe[k]*tot_scores_fe[k] for k in tot_scores_fe.keys()])/denom
        weighted_combined = np.sum([weights_combined[k]*tot_scores_combined[k] for k in tot_scores_combined.keys()])/denom
        weighted_asym_combined = np.sum([weights_asym_combined[k]*tot_scores_asym_combined[k] for k in tot_scores_asym_combined.keys()])/denom
        weighted_fe_lim = np.sum([weights_fe_lim[k]*tot_scores_fe_lim[k] for k in tot_scores_fe_lim.keys()])/denom
        weighted_asym = np.sum([weights_asym[k]*tot_scores_asym[k] for k in tot_scores_asym.keys()])/denom
        final_scores['gt'].append(weighted_gt)
        final_scores['fe'].append(weighted_fe)
        final_scores['asym'].append(weighted_asym)
        final_scores['combined'].append(weighted_combined)
        final_scores['asym_combined'].append(weighted_asym_combined)
        final_scores['fe_lim'].append(weighted_fe_lim)

        final_scores['fe_trajs'].append(np.array([tot_scores_fe[k] for k in tot_scores_fe.keys()]))
        final_scores['fe_traj_weights'].append(np.array([weights_fe[k] for k in tot_scores_fe.keys()]))
        final_scores['fe_idxs'].append(np.array([k for k in tot_scores_fe.keys()]))
        final_scores['asym_combined_trajs'].append(np.array([tot_scores_asym_combined[k] for k in tot_scores_asym_combined.keys()]))
        final_scores['asym_combined_traj_weights'].append(np.array([weights_asym_combined[k] for k in tot_scores_asym_combined.keys()]))
        final_scores['asym_combined_idxs'].append(np.array([k for k in tot_scores_asym_combined.keys()]))
        final_scores['gt_trajs'].append(np.array([tot_scores_gt[k] for k in tot_scores_gt.keys()]))
        final_scores['gt_traj_weights'].append(np.array([weights_gt[k] for k in tot_scores_gt.keys()]))
        final_scores['gt_idxs'].append(np.array([k for k in tot_scores_gt.keys()]))

        ratios = []
        for k in tot_scores_fe.keys():
            gt_score = tot_scores_gt[k]
            ac_score = tot_scores_asym_combined[k]
            ratio = ac_score / gt_score
            ratios.append(ratio)
        ratios = np.array(ratios)
        # or look at weighted_gt vs weighted_asym_combined

        # use top 0.5% and bottom 0.5% respectively
        if plot and (weighted_asym_combined > 48.86 or weighted_asym_combined < 2.16):
            tag = os.path.join(VISUAL_OUT_SUBDIR, scenario['scenario_id'])
            # TODO:
            # - probably also should save features/traj scores here for the select few scenes, as well as meta info
            # - also throw in the current rendering, may as well
            # - Consider also looking at anomolous examples of when asym_combined is much greater than GT
            plot_scene(scenario, tot_scores_gt, weighted_gt, interp_x, tot_scores_fe, weighted_fe, interp_y, weighted_asym_combined, tag)

    return final_scores

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='/av_shared_ssd/mtr_process_ssd')
    parser.add_argument('--split', type=str, default='training', choices=['training', 'validation', 'testing'])
    parser.add_argument('--num_scenarios', type=int, default=-1)
    parser.add_argument('--shards', type=int, default=10)
    parser.add_argument('--shard_idx', type=int, nargs='+',required=True)
    
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()

    VISUAL_OUT_DIR = 'out/scenario_id/vis'
    VISUAL_OUT_SUBDIR = os.path.join(VISUAL_OUT_DIR, f"{__file__.split('.')[0]}")
    os.makedirs(VISUAL_OUT_SUBDIR, exist_ok=True)

    agg_path = os.path.join(CACHE_DIR, args.split, 'features')
    gt_suffix = (f'_gt_shard{args.shard_idx[0]}_{args.shards}.npz')
    fe_suffix = (f'_fe_shard{args.shard_idx[0]}_{args.shards}.npz')
    asym_suffix = (f'_asym_shard{args.shard_idx[0]}_{args.shards}.npz')

    
    # Loading test shard 0 takes ~2 minutes, not *that* atrocious
    start = time.time()
    print(f"Loading individual features...")
    gt_individual_features_filepath = os.path.join(agg_path, f'individual_agg{gt_suffix}')
    gt_individual_features = np.load(gt_individual_features_filepath, allow_pickle=True)['arr_0']
    fe_individual_features_filepath = os.path.join(agg_path, f'individual_agg{fe_suffix}')
    fe_individual_features = np.load(fe_individual_features_filepath, allow_pickle=True)['arr_0']
    asym_individual_features_filepath = os.path.join(agg_path, f'individual_agg{asym_suffix}')
    asym_individual_features = np.load(asym_individual_features_filepath, allow_pickle=True)['arr_0']
    print(f"\t...process took {time.time() - start}")
    
    start = time.time()
    print(f"Loading interaction features...")
    gt_interaction_features_filepath = os.path.join(agg_path, f'interaction_agg{gt_suffix}')
    gt_interaction_features = np.load(gt_interaction_features_filepath, allow_pickle=True)['arr_0']
    fe_interaction_features_filepath = os.path.join(agg_path, f'interaction_agg{fe_suffix}')
    fe_interaction_features = np.load(fe_interaction_features_filepath, allow_pickle=True)['arr_0']
    asym_interaction_features_filepath = os.path.join(agg_path, f'interaction_agg{asym_suffix}')
    asym_interaction_features = np.load(asym_interaction_features_filepath, allow_pickle=True)['arr_0']
    print(f"\t...process took {time.time() - start}")
    
    start = time.time()
    print(f"Loading infos...")
    cache_path = os.path.join(CACHE_DIR, args.split)
    gt_infos = load_infos(base_path=args.base_path, cache_path=cache_path, split=args.split, 
        shard_idx=args.shard_idx, shards=args.shards, num_scenarios=args.num_scenarios, 
        load_lane_cache=False, load_conflict_points=False,
        load_type='gt', load_cluster_anomaly=False)
    fe_infos = load_infos(base_path=args.base_path, cache_path=cache_path, split=args.split, 
        shard_idx=args.shard_idx, shards=args.shards, num_scenarios=args.num_scenarios, 
        load_lane_cache=False, load_conflict_points=False,
        load_type='fe', load_cluster_anomaly=False)
    print(f"\t...process took {time.time() - start}")

    msg = f'Processing {args.split} split scenarios...'
    start = time.time()
    print(f"\tProcessing took {time.time() - start} seconds.")
    scores = compute_score(gt_individual_features, gt_interaction_features, gt_infos,
                          fe_individual_features, fe_interaction_features, fe_infos, 
                          asym_individual_features, asym_interaction_features, 
                          plot=args.plot)

    # Unpack and save 
    # Important for visualization purposes to not necessarily invoke this, unless you're certain it doesn't change anything
    if args.cache:
        CACHE_SUBDIR = os.path.join(cache_path, 'feature_scores')
        os.makedirs(CACHE_SUBDIR, exist_ok=True)
        suffix = (f'_shard{args.shard_idx[0]}_{args.shards}.npz')
        score_data = pd.DataFrame(scores)

# breakpoint()
# score_data = score_data[['gt', 'fe', 'combined', 'asym', 'asym_combined']]
# score_data.columns = ['Ground Truth', 'Future Extrapolated', 'Combined', 'Asymmetric', 'Asymmetric Combined']
# BIGGER_SIZE = 24
# plt.clf()
# plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=BIGGER_SIZE-4)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=BIGGER_SIZE-4)    # fontsize of the tick labels
# plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# sns.set_palette(sns.color_palette('tab10'))
# # sns.kdeplot(data=score_data['Ground Truth'], linewidth=3, bw_adjust=2, common_norm=False, label='Ground Truth', cut=0, fill=True, alpha=0.15)
# # sns.kdeplot(data=score_data['Future Extrapolated'], linewidth=3, bw_adjust=2, common_norm=False, label='Future Extrapolated', cut=0, fill=True, alpha=0.15)
# # sns.kdeplot(data=score_data['Combined'], linewidth=3, bw_adjust=2, common_norm=False, label='Combined', cut=0, fill=True, alpha=0.15)
# # sns.kdeplot(data=score_data['Asymmetric'], linewidth=3, bw_adjust=2, common_norm=False, label='Asymmetric', cut=0, fill=True, alpha=0.15)
# # sns.kdeplot(data=score_data['Asymmetric Combined'], linewidth=3, bw_adjust=2, common_norm=False, label='Asymmetric Combined', cut=0, fill=True, alpha=0.15)
# sns.kdeplot(data=score_data['Ground Truth'], linewidth=3, bw_adjust=2, common_norm=False, label='Ground Truth', cut=0,  )
# sns.kdeplot(data=score_data['Future Extrapolated'], linewidth=3, bw_adjust=2, common_norm=False, label='Future Extrapolated', cut=0,  )
# sns.kdeplot(data=score_data['Combined'], linewidth=3, bw_adjust=2, common_norm=False, label='Combined', cut=0,  )
# sns.kdeplot(data=score_data['Asymmetric'], linewidth=3, bw_adjust=2, common_norm=False, label='Asymmetric', cut=0,  )
# sns.kdeplot(data=score_data['Asymmetric Combined'], linewidth=3, bw_adjust=2, common_norm=False, label='Asymmetric Combined', cut=0,  )
# plt.xlabel('Scores')
# plt.ylabel('Density')
# plt.tight_layout()
# plt.legend()
# plt.savefig('scores_0.png', bbox_inches='tight', dpi=300)

        sns.kdeplot(data=score_data)
        plt.xlabel('Scores')
        plt.savefig(os.path.join(CACHE_SUBDIR, f'scores{suffix.split(".npz")[0]}_dist.png'))
        
        cache_filepath = os.path.join(CACHE_SUBDIR, f'scores{suffix}')
        with open(cache_filepath, 'wb') as f:
            np.savez(f, scores) 
        print(score_data.describe())
        print('99.5%', np.percentile(score_data.asym_combined, 99.5))
        print('0.5%', np.percentile(score_data.asym_combined, 0.5))

    print("Done.")