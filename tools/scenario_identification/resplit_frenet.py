import pickle as pkl
import os
import shutil
import pdb
import numpy as np

from tqdm import tqdm
from natsort import natsorted
from operator import itemgetter
from collections import deque
import argparse
import glob


base = os.path.expanduser('/av_shared_ssd/mtr_process_ssd')
train_base = f'{base}/new_processed_scenarios_training'
val_base = f'{base}/new_processed_scenarios_validation'
test_base = f'{base}/new_processed_scenarios_testing'
train_meta = f'{base}/new_processed_scenarios_training_infos.pkl'
val_meta = f'{base}/new_processed_scenarios_val_infos.pkl'
test_meta = f'{base}/new_processed_scenarios_test_infos.pkl'

frenet_train_meta = f'{base}/frenet_789_processed_scenarios_training_infos.pkl'
frenet_val_meta = f'{base}/frenet_789_processed_scenarios_val_infos.pkl'
frenet_test_meta = f'{base}/frenet_789_processed_scenarios_test_infos.pkl'

frenetp_train_labels = '/av_shared_ssd/scenario_id/cache/training/frenetp_clustering/labels_k-10_c-2_interp_shard0-9_10.pkl'
frenetp_val_labels = '/av_shared_ssd/scenario_id/cache/validation/frenetp_clustering/labels_k-10_c-2_interp_shard0-9_10.pkl'
frenetp_test_labels = '/av_shared_ssd/scenario_id/cache/testing/frenetp_clustering/labels_k-10_c-2_interp_shard0-9_10.pkl'

frenet_base = os.path.expanduser('~/monet_shared/shared/scenario_identification/cache/%s/frenet')

score_base = '/av_shared_ssd/scenario_id/cache/%s/feature_scores'
scores_train = score_base % 'training'
scores_val = score_base % 'validation'
scores_test = score_base % 'testing'

# For train/val split
splits = [0.80, 0.2]
def assign_label(cluster_label, rand_state, test_clusters):
    if cluster_label in test_clusters:
        return 'test'
    return 'train' if rand_state.rand() < splits[0] else 'val'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_clusters', type=str, default='789')
    parser.add_argument('--key', type=str, default='asym_combined', choices=['asym_combined'])
    parser.add_argument('--save_score', action='store_true')
    args = parser.parse_args()
    
    test_clusters = natsorted(list(set([x for x in args.test_clusters])))
    out_tag = ''.join(test_clusters)
    frenet_train_meta = frenet_train_meta.replace('789', out_tag)
    frenet_val_meta = frenet_val_meta.replace('789', out_tag)
    frenet_test_meta = frenet_test_meta.replace('789', out_tag)
    test_clusters = [int(x) for x in test_clusters]
    for x in test_clusters:
        assert 0 <= x <= 9, 'unsupported x val'

    save_score = bool(args.save_score)
    score_suffix = '_extra_processed' if save_score else '_processed'
    frenet_train_meta = frenet_train_meta.replace('_processed', score_suffix)
    frenet_val_meta = frenet_val_meta.replace('_processed', score_suffix)
    frenet_test_meta = frenet_test_meta.replace('_processed', score_suffix)
    if save_score:
        frenet_train_meta = frenet_train_meta.replace('extra', f'{args.key}_extra')
        frenet_val_meta = frenet_val_meta.replace('extra', f'{args.key}_extra')
        frenet_test_meta = frenet_test_meta.replace('extra', f'{args.key}_extra')

    with open(train_meta, 'rb') as f:
        train_metas = pkl.load(f)
    with open(val_meta, 'rb') as f:
        val_metas = pkl.load(f)
    with open(test_meta, 'rb') as f:
        test_metas = pkl.load(f)
    # Load meta pickle things
    train_metas = train_metas[::5]

    split_seed = 42
    rand_state = np.random.RandomState(split_seed)

    # The sorting is very important
    train_scores = natsorted(glob.glob(os.path.join(scores_train, '*.npz')))
    val_scores = natsorted(glob.glob(os.path.join(scores_val, '*.npz')))
    test_scores = natsorted(glob.glob(os.path.join(scores_test, '*.npz')))
    assert len(train_scores) == 10 and len(val_scores) == 10 and len(test_scores) == 10, 'Unexpected number of files'
    
    if args.key in ['gt', 'fe', 'combined', 'asym', 'asym_combined']:
        print(f'Loading "{args.key}" score from {scores_train}/*.npz')
        all_trains = []
        for train_file in train_scores:
            all_trains.extend(np.load(train_file, allow_pickle=True)['arr_0'].item()[args.key])
        print(f'Loading "{args.key}" score from {scores_val}/*.npz')
        all_vals = []
        for val_file in val_scores:
            all_vals.extend(np.load(val_file, allow_pickle=True)['arr_0'].item()[args.key])
        print(f'Loading "{args.key}" score from {scores_test}/*.npz')
        all_tests = []
        for test_file in test_scores:
            all_tests.extend(np.load(test_file, allow_pickle=True)['arr_0'].item()[args.key])
    elif args.key == 'plus':
        all_trains = []
        for train_file in train_scores:
            gt = np.load(train_file, allow_pickle=True)['arr_0'].item()['gt']
            fe = np.load(train_file, allow_pickle=True)['arr_0'].item()['fe']
            all_trains.extend([x + np.sqrt(y) for x, y in zip(gt, fe)])
        all_vals = []
        for val_file in val_scores:
            gt = np.load(val_file, allow_pickle=True)['arr_0'].item()['gt']
            fe = np.load(val_file, allow_pickle=True)['arr_0'].item()['fe']
            all_vals.extend([x + np.sqrt(y) for x, y in zip(gt, fe)])
        all_tests = []
        for test_file in test_scores:
            gt = np.load(test_file, allow_pickle=True)['arr_0'].item()['gt']
            fe = np.load(test_file, allow_pickle=True)['arr_0'].item()['fe']
            all_tests.extend([x + np.sqrt(y) for x, y in zip(gt, fe)])
    elif args.key == 'plus_cond':
        all_trains = []
        for train_file in train_scores:
            gt = np.load(train_file, allow_pickle=True)['arr_0'].item()['gt']
            fe = np.load(train_file, allow_pickle=True)['arr_0'].item()['fe']
            all_trains.extend([x + np.sqrt(y if y > x else x) for x, y in zip(gt, fe)])
        all_vals = []
        for val_file in val_scores:
            gt = np.load(val_file, allow_pickle=True)['arr_0'].item()['gt']
            fe = np.load(val_file, allow_pickle=True)['arr_0'].item()['fe']
            all_vals.extend([x + np.sqrt(y if y > x else x) for x, y in zip(gt, fe)])
        all_tests = []
        for test_file in test_scores:
            gt = np.load(test_file, allow_pickle=True)['arr_0'].item()['gt']
            fe = np.load(test_file, allow_pickle=True)['arr_0'].item()['fe']
            all_tests.extend([x + np.sqrt(y if y > x else x) for x, y in zip(gt, fe)])

    all_scores = all_trains + all_vals + all_tests
    threshold_score = np.percentile(all_scores, 80)

    print(f'Loading FE score from {scores_train}/*.npz')
    fe_trains = []
    for train_file in train_scores:
        fe_trains.extend(np.load(train_file, allow_pickle=True)['arr_0'].item()['fe'])
    print(f'Loading FE score from {scores_val}/*.npz')
    fe_vals = []
    for val_file in val_scores:
        fe_vals.extend(np.load(val_file, allow_pickle=True)['arr_0'].item()['fe'])
    print(f'Loading FE score from {scores_test}/*.npz')
    fe_tests = []
    for test_file in test_scores:
        fe_tests.extend(np.load(test_file, allow_pickle=True)['arr_0'].item()['fe'])
    
    print(f'Loading traj infos from {scores_train}/*.npz')
    asym_combined_trajs_trains = []
    asym_combined_traj_weights_trains = []
    asym_combined_idxs_trains = []
    fe_trajs_trains = []
    fe_traj_weights_trains = []
    fe_idxs_trains = []
    for train_file in train_scores:
        in_file = np.load(train_file, allow_pickle=True)['arr_0'].item()
        asym_combined_trajs_trains.extend(in_file['asym_combined_trajs'])
        asym_combined_traj_weights_trains.extend(in_file['asym_combined_traj_weights'])
        asym_combined_idxs_trains.extend(in_file['asym_combined_idxs'])
        fe_trajs_trains.extend(in_file['fe_trajs'])
        fe_traj_weights_trains.extend(in_file['fe_traj_weights'])
        fe_idxs_trains.extend(in_file['fe_idxs'])
    print(f'Loading traj infos from {scores_val}/*.npz')
    asym_combined_trajs_vals = []
    asym_combined_traj_weights_vals = []
    asym_combined_idxs_vals = []
    fe_trajs_vals = []
    fe_traj_weights_vals = []
    fe_idxs_vals = []
    for val_file in val_scores:
        in_file = np.load(val_file, allow_pickle=True)['arr_0'].item()
        asym_combined_trajs_vals.extend(in_file['asym_combined_trajs'])
        asym_combined_traj_weights_vals.extend(in_file['asym_combined_traj_weights'])
        asym_combined_idxs_vals.extend(in_file['asym_combined_idxs'])
        fe_trajs_vals.extend(in_file['fe_trajs'])
        fe_traj_weights_vals.extend(in_file['fe_traj_weights'])
        fe_idxs_vals.extend(in_file['fe_idxs'])
    print(f'Loading traj infos from {scores_test}/*.npz')
    asym_combined_trajs_tests = []
    asym_combined_traj_weights_tests = []
    asym_combined_idxs_tests = []
    fe_trajs_tests = []
    fe_traj_weights_tests = []
    fe_idxs_tests = []
    for test_file in test_scores:
        in_file = np.load(test_file, allow_pickle=True)['arr_0'].item()
        asym_combined_trajs_tests.extend(in_file['asym_combined_trajs'])
        asym_combined_traj_weights_tests.extend(in_file['asym_combined_traj_weights'])
        asym_combined_idxs_tests.extend(in_file['asym_combined_idxs'])
        fe_trajs_tests.extend(in_file['fe_trajs'])
        fe_traj_weights_tests.extend(in_file['fe_traj_weights'])
        fe_idxs_tests.extend(in_file['fe_idxs'])

    train_metas_out = []
    val_metas_out = []
    test_metas_out = []
    for meta, label_path, original_base, scores, fe_scores, \
        asym_combined_trajs, asym_combined_traj_weights, asym_combined_idxs, \
        fe_trajs, fe_traj_weights, fe_idxs, \
         in zip([train_metas, val_metas, test_metas],
                                               [frenetp_train_labels, frenetp_val_labels, frenetp_test_labels],
                                               [train_base, val_base, test_base],
                                               [all_trains, all_vals, all_tests],
                                               [fe_trains, fe_vals, fe_tests],
                                               [asym_combined_trajs_trains, asym_combined_trajs_vals, asym_combined_trajs_tests],
                                               [asym_combined_traj_weights_trains, asym_combined_traj_weights_vals, asym_combined_traj_weights_tests],
                                               [asym_combined_idxs_trains, asym_combined_idxs_vals, asym_combined_idxs_tests],
                                               [fe_trajs_trains, fe_trajs_vals, fe_trajs_tests],
                                               [fe_traj_weights_trains, fe_traj_weights_vals, fe_traj_weights_tests],
                                               [fe_idxs_trains, fe_idxs_vals, fe_idxs_tests]):
        labels = np.load(label_path, allow_pickle=True)['full_traj_labels']

        assert len(labels) == len(meta) == len(scores) == len(fe_scores), 'All lengths must match'
        for (scenario_id, cluster_label), meta_info, score, fe_score, \
            asym_combined_traj, asym_combined_traj_weight, asym_combined_idx, \
            fe_traj, fe_traj_weight, fe_idx, in tqdm(zip(labels.items(), meta, scores, fe_scores, \
                                                    asym_combined_trajs, asym_combined_traj_weights, asym_combined_idxs, \
                                                    fe_trajs, fe_traj_weights, fe_idxs),
                                                    f'Processing scenarios...', total=len(labels), dynamic_ncols=True):
            assert scenario_id.split('sample_')[-1] == meta_info['scenario_id'], 'Mismatch in ordering'
            new_label = assign_label(cluster_label, rand_state, test_clusters)
            scenario_path = f'{original_base}/{scenario_id}.pkl'
            if save_score:
                meta_info['score'] = score
                # TODO: also save some features here? This'd be the spot to do it...
                meta_info['fe_score'] = fe_score
                base_idxs = meta_info['tracks_to_predict']['track_index']
                ac_real_idxs = [np.where(asym_combined_idx == base_idx)[0].item() for base_idx in base_idxs]
                fe_real_idxs = [np.where(fe_idx == base_idx)[0].item() for base_idx in base_idxs]
                ac_traj_score = asym_combined_traj[ac_real_idxs]
                fe_traj_score = fe_traj[fe_real_idxs]
                meta_info['traj_scores_asym_combined'] = ac_traj_score
                meta_info['traj_scores_fe'] = fe_traj_score

            if new_label == 'train':
                train_metas_out.append(meta_info)
            elif new_label == 'val':
                val_metas_out.append(meta_info)
            else:
                test_metas_out.append(meta_info)

    with open(frenet_train_meta, 'wb') as f:
        pkl.dump(train_metas_out, f)
    with open(frenet_val_meta, 'wb') as f:
        pkl.dump(val_metas_out, f)
    with open(frenet_test_meta, 'wb') as f:
        pkl.dump(test_metas_out, f)
    print(len(train_metas_out), len(val_metas_out), len(test_metas_out))
    