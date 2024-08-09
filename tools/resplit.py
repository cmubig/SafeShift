import pickle as pkl
import os
import shutil
import pdb
import numpy as np

from tqdm import tqdm
from natsort import natsorted
from operator import itemgetter


base = os.path.expanduser('~/waymo/mtr_process')
train_base = f'{base}/processed_scenarios_training'
val_base = f'{base}/processed_scenarios_validation'
train_meta = f'{base}/processed_scenarios_training_infos.pkl'
val_meta = f'{base}/processed_scenarios_val_infos.pkl'

new_train_base = f'{base}/new_processed_scenarios_training'
new_val_base = f'{base}/new_processed_scenarios_validation'
new_test_base = f'{base}/new_processed_scenarios_testing'
new_train_meta = f'{base}/new_processed_scenarios_training_infos.pkl'
new_val_meta = f'{base}/new_processed_scenarios_val_infos.pkl'
new_test_meta = f'{base}/new_processed_scenarios_test_infos.pkl'


splits = [0.85, 0.075, 0.075]
if __name__ == '__main__':
    os.makedirs(new_train_base, exist_ok=True)
    os.makedirs(new_val_base, exist_ok=True)
    os.makedirs(new_test_base, exist_ok=True)

    split_seed = 42
    rand_state = np.random.RandomState(split_seed)
    train_inputs = [(x, f'{train_base}/{x}') for x in os.listdir(train_base)]
    val_inputs =  [(x, f'{val_base}/{x}') for x in os.listdir(val_base)]
    with open(train_meta, 'rb') as f:
        train_metas = pkl.load(f)
    with open(val_meta, 'rb') as f:
        val_metas = pkl.load(f)
    # Load meta pickle things
    input_scenarios = natsorted(train_inputs) + natsorted(val_inputs)
    input_metas = natsorted(train_metas, key=itemgetter(*['scenario_id'])) + \
                  natsorted(val_metas, key=itemgetter(*['scenario_id']))

    train_metas = []
    val_metas = []
    test_metas = []
    for (scenario, path), input_meta in tqdm(zip(input_scenarios, input_metas), 'Processing scenarios...', total=len(input_scenarios)):
        decision = rand_state.rand()
        if decision < splits[0]:
            shutil.copy(path, f'{new_train_base}/{scenario}')
            train_metas.append(input_meta)
        elif decision < splits[0] + splits[1]:
            shutil.copy(path, f'{new_val_base}/{scenario}')
            val_metas.append(input_meta)
        else:
            assert decision < splits[0] + splits[1] + splits[2], 'Invalid rand'
            shutil.copy(path, f'{new_test_base}/{scenario}')
            test_metas.append(input_meta)
    with open(new_train_meta, 'wb') as f:
        pkl.dump(train_metas, f)
    with open(new_val_meta, 'wb') as f:
        pkl.dump(val_metas, f)
    with open(new_test_meta, 'wb') as f:
        pkl.dump(test_metas, f)
    