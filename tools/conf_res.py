import argparse
import pdb
import os
import pickle as pkl
import logging
import numpy as np
import torch
import pingouin as pg

from tqdm import tqdm
from matplotlib import pyplot as plt

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

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--path', choices=['a', 'b', 'c', 'd', 'e'], default='a', help='Which file to use')
    # args = parser.parse_args()
    choices = ['a', 'b', 'c', 'd', 'e']
    base = '../output/new_waymo/mtr+20p_%s/default/eval/eval_with_train/result.pkl'
    results_paths = [base % x for x in choices]

    all_results = []
    for results_path in results_paths:
        with open(results_path, 'rb') as f:
            results = pkl.load(f)
        all_results.append(results)
    
    all_scores = []
    all_ades = []
    for paired_result in tqdm(zip(*all_results), 'Processing scenarios', total=len(all_results[0])):
        assert len(set([result['scenario_id'] for result in paired_result])) == 1, 'Mismatched scenario_id'
        assert len(set([result['object_id'] for result in paired_result])) == 1, 'Mismatched object_id'
        # for result in paired_result:
        #     scenario_id = result['scenario_id']
        #     avg_results = waymo_evaluation_explore([result], num_modes_for_eval=6)
        #     result['avg_results'] = avg_results
        ensemble_result = joint_ensemble(paired_result)
        gt_xy = ensemble_result['gt_trajs'][-80:, :2]
        gt_valid = ensemble_result['gt_trajs'][-80:, 9]

        gt_xy = gt_xy[gt_valid == 1]
        pred_xy = ensemble_result['pred_trajs'][:, gt_valid == 1]
        ade = np.mean(np.sqrt(np.sum((pred_xy - gt_xy)**2, axis=-1)), axis=-1)
        scores = ensemble_result['pred_scores']
        all_scores.extend(scores)
        all_ades.extend(ade)

    # ADE and FDE are simply averaged over the valid points in GT (if at least 1)
    import pdb; pdb.set_trace()
    plt.clf()
    x_data = all_scores
    y_data = all_ades
    plt.xlabel(f'Confidence Scores')
    plt.ylabel(f'Agent ADEs')
    r = pg.corr(x_data, y_data).r.item() 
    plt.scatter(x_data, y_data, s=4)
    plt.annotate(f'r = {r:.3f}', xy=(0.7, 0.9), xycoords='axes fraction')
    plt.savefig(f'conf_ade_corr.png')
    #avg_results = waymo_evaluation_explore(ensemble_results, num_modes_for_eval=len(ensemble_results[0]['pred_trajs']))
