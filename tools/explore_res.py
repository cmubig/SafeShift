import argparse
import pdb
import os
import pickle as pkl
import logging

from tqdm import tqdm

from mtr.datasets.waymo.waymo_eval import waymo_evaluation_explore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', choices=['a', 'b', 'c', 'd', 'e'], default='a', help='Which file to use')
    args = parser.parse_args()
    results_path = f'../output/new_waymo/mtr+20p_{args.path}/default/eval/eval_with_train/result.pkl'
    logging.disable(logging.CRITICAL)

    with open(results_path, 'rb') as f:
        results = pkl.load(f)
    
    for result in tqdm(results, 'Processing scenarios'):
        scenario_id = result['scenario_id']
        avg_results = waymo_evaluation_explore([result], num_modes_for_eval=6)
        result['avg_results'] = avg_results
    results_out_path = results_path.replace('result.pkl', 'result_explore.pkl')
    with open(results_out_path, 'wb') as f:
        pkl.dump(results, f)
