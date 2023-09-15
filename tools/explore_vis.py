import argparse
import pdb
import os
import pickle as pkl
import pandas as pd
import pingouin as pg

from tqdm import tqdm
from matplotlib import pyplot as plt

from mtr.datasets.waymo.waymo_eval import waymo_evaluation_explore


def get_data(results_path):
    with open(results_path, 'rb') as f:
        results = pkl.load(f)
    
    min_ades = []
    min_fdes = []
    miss_rates = []
    overlap_rates = []
    mAPs = []
    for result in results:
        assert result['object_type'] in ['TYPE_PEDESTRIAN', 'TYPE_CYCLIST', 'TYPE_VEHICLE'], 'Unknown type'
        if result['object_type'] == 'TYPE_PEDESTRIAN':
            key = 'PEDESTRIAN'
        elif result['object_type'] == 'TYPE_CYCLIST':
            key = 'CYCLIST'
        elif result['object_type'] == 'TYPE_VEHICLE':
            key = 'VEHICLE'
        min_ades.append(result['avg_results'][f'minADE - {key}'])
        min_fdes.append(result['avg_results'][f'minFDE - {key}'])
        miss_rates.append(result['avg_results'][f'MissRate - {key}'])
        overlap_rates.append(result['avg_results'][f'OverlapRate - {key}'])
        mAPs.append(result['avg_results'][f'mAP - {key}'])

    data = {
        'scenario_id': [result['scenario_id'] for result in results],
        'object_type': [result['object_type'] for result in results],
        'object_id': [result['object_id'] for result in results],
        'min_ade': min_ades, 'min_fde': min_fdes, 'miss_rate': miss_rates, 'overlap_rate': overlap_rates, 'mAP': mAPs
    }
    data = pd.DataFrame(data).sort_values(['scenario_id', 'object_id'])
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    choices = ['a', 'b', 'c', 'd', 'e']
    parser.add_argument('--path', choices=[*choices, 'all'], default='all', help='Which file to use')
    args = parser.parse_args()
    if args.path == 'all':
        labels = [*choices]
    else:
        labels = [args.path]
    base = '../output/new_waymo/mtr+20p_%s/default/eval/eval_with_train/result_explore.pkl'
    results_paths = [base % label for label in labels]

    #data = data[data.object_type == 'TYPE_VEHICLE']
    datas = []
    for results_path in results_paths:
        data = get_data(results_path)
        datas.append(data)

    for data, label in zip(datas, labels):
        scene_ade = data.groupby('scenario_id').min_ade.mean().to_numpy()
        plt.hist(scene_ade, bins=100, alpha=0.4, label=label)
    plt.legend()
    plt.savefig(f'tmp_ade.png')

    plt.clf()
    for i in range(len(datas)):
        for j in range(len(datas)):
            if j <= i:
                continue
            plt.clf()
            x_data = datas[i].groupby('scenario_id').min_ade.mean().to_numpy()
            y_data = datas[j].groupby('scenario_id').min_ade.mean().to_numpy()
            x_name = chr(ord('a') + i) 
            y_name = chr(ord('a') + j) 
            plt.xlabel(f'MTR-{x_name.upper()} Scenario ADE')
            plt.ylabel(f'MTR-{y_name.upper()} Scenario ADE')
            r = pg.corr(x_data, y_data).r.item() 
            plt.scatter(x_data, y_data, s=4)
            plt.annotate(f'r = {r:.3f}', xy=(0.7, 0.9), xycoords='axes fraction')
            plt.savefig(f'tmp_corr_{x_name}{y_name}.png')
    import pdb; pdb.set_trace()

    