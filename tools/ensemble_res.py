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
import contextlib
from joblib import Parallel, delayed

from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
import dtw
from natsort import natsorted

# Force using CPU instead of GPU, since calling waymo_eval so often...
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from mtr.datasets.waymo.waymo_eval import waymo_evaluation_explore, waymo_evaluation
from mtr.utils.motion_utils import batch_nms
from mtr.config import cfg, cfg_from_yaml_file
from mtr.utils import common_utils

def joint_ensemble(paired_result):
    ret = {}
    ret['scenario_id'] = paired_result[0]['scenario_id']
    ret['object_id'] = paired_result[0]['object_id']
    ret['object_type'] = paired_result[0]['object_type']
    ret['gt_trajs'] = paired_result[0]['gt_trajs'].copy()
    ret['track_index_to_predict'] = paired_result[0]['track_index_to_predict'].copy()
    # Now, ensemble together pred_trajs and pred_scores
    pred_scores = []
    pred_trajs = []
    for result in paired_result:
        pred_scores.append(result['pred_scores'])
        pred_trajs.append(result['pred_trajs'])
    ret['pred_scores'] = np.concatenate(pred_scores)
    ret['pred_trajs'] = np.concatenate(pred_trajs)
    return ret

def conf_ensemble(paired_result):
    ret = {}
    ret['scenario_id'] = paired_result[0]['scenario_id']
    ret['object_id'] = paired_result[0]['object_id']
    ret['object_type'] = paired_result[0]['object_type']
    ret['gt_trajs'] = paired_result[0]['gt_trajs'].copy()
    ret['track_index_to_predict'] = paired_result[0]['track_index_to_predict'].copy()
    # Now, ensemble together pred_trajs and pred_scores
    pred_scores = []
    pred_trajs = []
    max_conf = []
    for result in paired_result:
        pred_scores.append(result['pred_scores'])
        pred_trajs.append(result['pred_trajs'])
        max_conf.append(result['pred_scores'].max())

    best_idx = np.argmax(max_conf)
    ret['pred_scores'] = pred_scores[best_idx]
    ret['pred_trajs'] = pred_trajs[best_idx]
    # ret['pred_scores'] = np.concatenate(pred_scores)
    # ret['pred_trajs'] = np.concatenate(pred_trajs)
    # ind = np.flip(np.argpartition(ret['pred_scores'], -6)[-6:])
    # ret['pred_scores'] = ret['pred_scores'][ind]
    # ret['pred_trajs'] = ret['pred_trajs'][ind]
    return ret

# Use GT Trajs to select the best ones, provide an upper bound
def cheat_ensemble(paired_result, label=None):
    ret = {}
    ret['scenario_id'] = paired_result[0]['scenario_id']
    ret['object_id'] = paired_result[0]['object_id']
    ret['object_type'] = paired_result[0]['object_type']
    ret['gt_trajs'] = paired_result[0]['gt_trajs'].copy()
    ret['track_index_to_predict'] = paired_result[0]['track_index_to_predict'].copy()
    # Now, ensemble together pred_trajs and pred_scores
    if label is None:
        pred_scores = []
        pred_trajs = []
        mAPs = []
        possible_keys = ['mAP - VEHICLE', 'mAP - PEDESTRIAN', 'mAP - CYCLIST']
        for result in paired_result:
            avg_results = waymo_evaluation_explore([result], num_modes_for_eval=6)
            mAP = 0
            n_val = 0
            for key in possible_keys:
                if avg_results[key] != -1:
                    mAP = avg_results[key]
                    n_val += 1
            assert n_val == 1, 'Key not found'
            mAPs.append(mAP)
            pred_scores.append(result['pred_scores'])
            pred_trajs.append(result['pred_trajs'])
        best_idx = np.argmax(mAPs)
        ret['pred_scores'] = pred_scores[best_idx]
        ret['pred_trajs'] = pred_trajs[best_idx]
    else:
        best_idx = label['mAP']['idx']
        ret['pred_scores'] = paired_result[best_idx]['pred_scores']
        ret['pred_trajs'] = paired_result[best_idx]['pred_trajs']
    return ret

def nms_ensemble(paired_result):
    ret = {}
    ret['scenario_id'] = paired_result[0]['scenario_id']
    ret['object_id'] = paired_result[0]['object_id']
    ret['object_type'] = paired_result[0]['object_type']
    ret['gt_trajs'] = paired_result[0]['gt_trajs'].copy()
    ret['track_index_to_predict'] = paired_result[0]['track_index_to_predict'].copy()
    # Now, ensemble together pred_trajs and pred_scores
    pred_scores = []
    pred_trajs = []
    for result in paired_result:
        pred_scores.append(result['pred_scores'])
        pred_trajs.append(result['pred_trajs'])
    ret['pred_scores'] = np.concatenate(pred_scores)
    ret['pred_trajs'] = np.concatenate(pred_trajs)

    # From MTR-A Paper
    max_conf_idx = np.argmax(ret['pred_scores'])
    max_conf_traj = ret['pred_trajs'][max_conf_idx]
    traj_length = np.sqrt((max_conf_traj[-1][0] - max_conf_traj[0][0])**2 + \
                          (max_conf_traj[-1][1] - max_conf_traj[0][1])**2)
    delta = min(3.5, max(2.5, (traj_length - 10)/(50-10)*1.5 + 2.5))
    # TODO: Refine NMS stuff here...
    ret_trajs, ret_scores, _ = batch_nms(torch.tensor(ret['pred_trajs'][np.newaxis, ...]), 
                                                torch.tensor(ret['pred_scores'][np.newaxis, ...]), 
                                                dist_thresh=delta, num_ret_modes=6)
    ret['pred_scores'] = ret_scores.numpy()[0]
    ret['pred_trajs'] = ret_trajs.numpy()[0]
    return ret

def kmeans_ensemble(paired_result):
    ret = {}
    ret['scenario_id'] = paired_result[0]['scenario_id']
    ret['object_id'] = paired_result[0]['object_id']
    ret['object_type'] = paired_result[0]['object_type']
    ret['gt_trajs'] = paired_result[0]['gt_trajs'].copy()
    ret['track_index_to_predict'] = paired_result[0]['track_index_to_predict'].copy()
    # Now, ensemble together pred_trajs and pred_scores
    pred_scores = []
    pred_trajs = []
    for result in paired_result:
        pred_scores.append(result['pred_scores'])
        pred_trajs.append(result['pred_trajs'])
    merged_scores = np.concatenate(pred_scores)
    merged_trajs = np.concatenate(pred_trajs)
    # Cluster based on end points for now
    # Attempting to follow ensemble as described in Golfer
    # kmeans = KMeans(n_clusters=6, random_state=42, n_init='auto').fit(merged_trajs[:, -1, :])
    # cluster_scores = [[] for _ in range(6)]
    # cluster_trajs = [[] for _ in range(6)]
    # for i, (traj, score) in enumerate(zip(merged_trajs, merged_scores)):
    #     cluster_i = kmeans.labels_[i]
    #     cluster_scores[cluster_i].append(score)
    #     cluster_trajs[cluster_i].append(traj)
    # tot_score = np.sum([np.sum(x) for x in cluster_scores])
    # cluster_trajs = np.stack([np.stack(x).mean(axis=0) for x in cluster_trajs])
    # cluster_scores = np.array([np.sum(x)/tot_score for x in cluster_scores])

    kmeans = KMeans(n_clusters=6, random_state=42, n_init='auto').fit(merged_trajs.reshape(merged_trajs.shape[0], -1))
    cluster_scores = [[] for _ in range(6)]
    cluster_trajs = [[] for _ in range(6)]
    for i, (traj, score) in enumerate(zip(merged_trajs, merged_scores)):
        cluster_i = kmeans.labels_[i]
        cluster_scores[cluster_i].append(score)
        cluster_trajs[cluster_i].append(traj)
    tot_score = np.sum([np.sum(x) for x in cluster_scores])
    cluster_trajs = kmeans.cluster_centers_.reshape((6, *merged_trajs.shape[1:]))
    cluster_scores = np.array([np.sum(x)/tot_score for x in cluster_scores])

    ret['pred_scores'] = cluster_scores
    ret['pred_trajs'] = cluster_trajs
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', default='cfgs/meta/mtr+20p_initial.yaml', help='Which ensemble config to use')
    parser.add_argument('--extra_tag', default='default')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    dataset_cfg = cfg.DATA_CONFIG
    dataset_cfg.MODELS = sorted(dataset_cfg.MODELS, key=lambda x: x['ckpt'])
    model_short = hashlib.shake_256(json.dumps(cfg.DATA_CONFIG.MODELS, sort_keys=True).encode()).hexdigest(4)
    label_path = dataset_cfg.DATASET_PATH + '/' + model_short + '.pkl'
    label_path = str(cfg.ROOT_DIR) + '/' + label_path

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    os.makedirs(output_dir, exist_ok=True)

    import pdb; pdb.set_trace()
    log_file = output_dir / ('log_ensemble_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start logging**********************')

    if os.path.exists(label_path):
        # Read from cache:
        with open(label_path, 'rb') as f:
            labels = pkl.load(f)
            labels['train'] = natsorted(labels['train'], key=lambda x: x['scenario_id'])
            labels['val'] = natsorted(labels['val'], key=lambda x: x['scenario_id'])
            labels['test'] = natsorted(labels['test'], key=lambda x: x['scenario_id'])
    else:
        labels = None
        print('Labels not cached; will compute on the fly')
        #raise NotImplementedError('Still need to implement caching stuff')

    # ADE and FDE are simply averaged over the valid points in GT (if at least 1)
    def get_str(results):
        metric_results, result_format_str  = waymo_evaluation(results, num_modes_for_eval=len(results[0]['pred_trajs']))
        metric_result_str = '\n'
        for key in metric_results:
            metric_results[key] = metric_results[key]
            metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
        metric_result_str += '\n'
        metric_result_str += result_format_str
        return metric_result_str

    def filter_str(str):
        return '\n'.join(str.split('\n')[-7:-1])

    splits = ['train', 'val', 'test']
    all_results = {split: [] for split in splits}
    for model in tqdm(dataset_cfg.MODELS, desc='Loading results...'):
        for split in splits:
            res_path = str(cfg.ROOT_DIR) + '/' + model[f'{split}_results']
            with open(res_path, 'rb') as f:
                results = pkl.load(f)
            results = natsorted(results, key=lambda x: x['scenario_id'])
            all_results[split].append(results)
    if labels is not None:
        labels_dict = {}
        for split in splits:
            labels_split = labels[split]
            labels_dict[split] = {}
            for label in labels_split:
                scenario_id = label['scenario_id']
                object_id = label['object_id']
                if scenario_id in labels_dict[split]:
                    labels_dict[split][scenario_id][object_id] = label
                else:
                    labels_dict[split][scenario_id] = {object_id: label}
    
    out_strs = {}
    for split in splits:
        out_strs[split] = {}
        all_results_split = all_results[split]

        logger.info('*************************************')
        logger.info('-------'+split+'-------')

        for i, base_result in enumerate(all_results_split):
            out_strs[split][f'base_{i}'] = filter_str(get_str(base_result))
        for i in range(len(all_results_split)):
            base_key = f'base_{i}'
            logger.info(f'Base Results {i}:\n{out_strs[split][base_key]}')

        def process_cheat(paired_result, label):
            assert len(set([result['scenario_id'] for result in paired_result])) == 1, 'Mismatched scenario_id'
            assert len(set([result['object_id'] for result in paired_result])) == 1, 'Mismatched object_id'
            return cheat_ensemble(paired_result, label=label)
        def process_nms(paired_result):
            return nms_ensemble(paired_result)
        def process_conf(paired_result):
            return conf_ensemble(paired_result)
        def process_kmeans(paired_result):
            return kmeans_ensemble(paired_result)
        

        if labels is not None:
            all_labels = [labels_dict[split][paired_result[0]['scenario_id']][paired_result[0]['object_id']] \
                        for paired_result in zip(*all_results_split)]
        else:
            all_labels = [None for _ in zip(*all_results_split)]
        cheat_results = Parallel(n_jobs=1)(delayed(process_cheat)(paired_result, label) \
                                     for paired_result, label in tqdm(zip(zip(*all_results_split), all_labels), 'Processing scenarios', total=len(all_results_split[0])))
        kmeans_results = Parallel(n_jobs=4)(delayed(process_kmeans)(paired_result) \
                                     for paired_result in tqdm(zip(*all_results_split), 'Processing scenarios', total=len(all_results_split[0])))
        nms_results = Parallel(n_jobs=4)(delayed(process_nms)(paired_result) \
                                     for paired_result in tqdm(zip(*all_results_split), 'Processing scenarios', total=len(all_results_split[0])))
        conf_results = Parallel(n_jobs=4)(delayed(process_conf)(paired_result) \
                                     for paired_result in tqdm(zip(*all_results_split), 'Processing scenarios', total=len(all_results_split[0])))

        out_strs[split]['nms'] = filter_str(get_str(nms_results))
        out_strs[split]['conf'] = filter_str(get_str(conf_results))
        out_strs[split]['kmeans'] = filter_str(get_str(kmeans_results))
        out_strs[split]['cheat'] = filter_str(get_str(cheat_results))

        logger.info(f'NMS Ensemble Results:\n{out_strs[split]["nms"]}')
        logger.info(f'Conf Ensemble Results:\n{out_strs[split]["conf"]}')
        logger.info(f'KMeans Ensemble Results:\n{out_strs[split]["kmeans"]}')
        logger.info(f'Cheat Ensemble Results:\n{out_strs[split]["cheat"]}')
