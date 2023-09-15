# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import os
import numpy as np
from pathlib import Path
import pickle
import torch
import hashlib
import json
from tqdm import tqdm
from natsort import natsorted

from mtr.datasets.dataset import DatasetTemplate
from mtr.utils import common_utils
from mtr.config import cfg
from mtr.datasets.waymo.waymo_eval import waymo_evaluation_explore, waymo_evaluation


class MetaDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training=True, logger=None, split=None):
        super().__init__(dataset_cfg=dataset_cfg, training=training, logger=logger)
        # Sort models, to make sure hashing is consistent
        dataset_cfg.MODELS = sorted(dataset_cfg.MODELS, key=lambda x: x['ckpt'])
        model_short = hashlib.shake_256(json.dumps(cfg.DATA_CONFIG.MODELS, sort_keys=True).encode()).hexdigest(4)
        self.label_path = dataset_cfg.DATASET_PATH + '/' + model_short + '.pkl'
        self.label_path = str(cfg.ROOT_DIR) + '/' + self.label_path

        assert split in ['train', 'val', 'test'], 'Invalid split provided'
        logger.info(f'Processing data for {split} split')

        logger.info('Attempting to read cached labels at ' + self.label_path)
        if os.path.exists(self.label_path):
            # Read from cache:
            with open(self.label_path, 'rb') as f:
                self.labels = pickle.load(f)
                self.labels['train'] = natsorted(self.labels['train'], key=lambda x: x['scenario_id'])
                self.labels['val'] = natsorted(self.labels['val'], key=lambda x: x['scenario_id'])
                self.labels['test'] = natsorted(self.labels['test'], key=lambda x: x['scenario_id'])
        else:
            raise NotImplementedError('Still need to implement caching stuff')

        self.processed_path = self.label_path.replace('.pkl', '_processed.pkl')
        if os.path.exists(self.processed_path):
            self.logger.info(f'Loading processed pickle from {self.processed_path}')
            with open(self.processed_path, 'rb') as f:
                processed = pickle.load(f)
                self.labels = processed['labels']
                self.all_results = processed['all_results']
        else:
            splits = self.labels.keys()
            self.all_results = {split: [] for split in splits}
            for model in tqdm(dataset_cfg.MODELS, desc='Loading results...'):
                for split in splits:
                    res_path = str(cfg.ROOT_DIR) + '/' + model[f'{split}_results']
                    with open(res_path, 'rb') as f:
                        results = pickle.load(f)
                    results = natsorted(results, key=lambda x: x['scenario_id'])
                    self.all_results[split].append(results)
            self.logger.info('Loaded all results')
            assert len(set([len(x) for x in self.all_results['val']])) == 1, 'Mismatch in val lengths'
            assert len(set([len(x) for x in self.all_results['test']])) == 1, 'Mismatch in test lengths'
            assert len(set([len(x) for x in self.all_results['train']])) == 1, 'Mismatch in train lengths'

            if 'train' not in self.labels:
                train_split = 0.85
                split_random = np.random.RandomState(42)
                train_labels = []
                val_labels = []
                n_models = len(self.all_results['val'])
                all_train = [[] for _ in range(n_models)]
                all_val = [[] for _ in range(n_models)]
                for scenario_idx, scenario in tqdm(enumerate(self.labels['val']), 'Splitting train/val'):
                    if split_random.rand() < train_split:
                        train_labels.append(scenario)
                        for i in range(n_models):
                            to_append = self.all_results['val'][i][scenario_idx]
                            assert to_append['scenario_id'] == scenario['scenario_id'], 'Mismatch scenario ID'
                            all_train[i].append(to_append)
                    else:
                        val_labels.append(scenario)
                        for i in range(n_models):
                            to_append = self.all_results['val'][i][scenario_idx]
                            assert to_append['scenario_id'] == scenario['scenario_id'], 'Mismatch scenario ID'
                            all_val[i].append(to_append)
                self.all_results['val'] = all_val
                self.all_results['train'] = all_train
                self.labels['val'] = val_labels
                self.labels['train'] = train_labels

            with open(self.processed_path, 'wb') as f:
                processed = {'labels': self.labels, 'all_results': self.all_results}
                pickle.dump(processed, f)
            self.logger.info(f'Saved processed pickle to {self.processed_path}')

        self.logger.info(f'Total scenes, train/val/test: '\
                         f'{len(self.labels["train"])}/{len(self.labels["val"])}/{len(self.labels["test"])}')
        self.split = split
        self.labels = self.labels[split]
        self.all_results = self.all_results[split]
        self.all_results_dict = {}
        for result_idx in range(len(self.all_results[0])):
            results = [result[result_idx] for result in self.all_results]
            scenario_id = results[0]['scenario_id']
            object_id = results[0]['object_id']
            if scenario_id in self.all_results_dict:
                self.all_results_dict[scenario_id][object_id] = results
            else:
                self.all_results_dict[scenario_id] = {object_id: results}

    def __len__(self):
        return len(self.labels)

    def total_tracks(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {'label': self.labels[index], 'all_results': [self.all_results[i][index] for i in range(len(self.all_results))]}
    
    def generate_prediction_dicts(self, batch_preds, output_path=None):
        # Return unmodified
        return [{
            'batch_size': batch_preds['batch_size'],
            'input_dict': batch_preds['input_dict'],
            'batch_sample_count': batch_preds['batch_sample_count'],
            'pred_labels': batch_preds['pred_labels'].cpu()
        }]
    
    def evaluation(self, pred_dicts, output_path=None, model=None):
        # gt = torch.cat([x['input_dict']['label'].cpu() for x in pred_dicts])
        # preds = torch.cat([x['pred_labels']argmax(dim=-1).cpu() for x in pred_dicts])
        # return f'Accuracy: {acc}', {'Accuracy': acc}
        gt_single = torch.cat([x['input_dict']['label_single'].cpu() for x in pred_dicts])
        gt = torch.cat([x['input_dict']['label'].cpu() for x in pred_dicts])
        preds = torch.cat([x['pred_labels'].cpu() for x in pred_dicts])
        # both are N x 5

        # Missing mAP Loss (does not work, just greedily selects "best" on average of the 5 models)
        # best_mAP = gt.max(dim=1)[0]
        # missing_mAP = best_mAP.unsqueeze(-1) - gt
        # missing_mAP = (torch.nn.functional.softmax(preds, dim=1)*missing_mAP).sum().item()/len(gt)
        # eval_loss = missing_mAP

        # CrossEntropy Loss
        # best_indices = torch.zeros_like(gt)
        # best_indices[gt == gt.max(dim=1)[0].unsqueeze(-1)] = 1
        # target = best_indices/best_indices.sum(dim=1).unsqueeze(-1)
        # criterion = torch.nn.CrossEntropyLoss()
        # eval_loss = criterion(preds, target)

        # MSE Loss
        # eval_loss = torch.nn.functional.mse_loss(preds, gt).item()

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            eval_loss = model.module.get_loss(preds, gt).item()
        else:
            eval_loss = model.get_loss(preds, gt).item()

        selection = preds.argmax(dim=-1)
        pred_scenario_id = np.concatenate([x['input_dict']['scenario_id'] for x in pred_dicts])
        pred_object_id = np.concatenate([x['input_dict']['object_id'] for x in pred_dicts])
        results = [self.all_results_dict[scenario_id][object_id][model_idx] for model_idx, scenario_id, object_id in zip(selection, pred_scenario_id, pred_object_id)]

        metric_results, result_format_str  = waymo_evaluation(results, num_modes_for_eval=len(results[0]['pred_trajs']))
        metric_result_str = '\n'
        for key in metric_results:
            metric_results[key] = metric_results[key]
            metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
        metric_result_str += '\n'
        metric_result_str += result_format_str
        mAP = [x for x in metric_result_str.split('\n') if x.startswith('mAP: 0.')][0]
        mAP = float(mAP.split(': ')[-1].strip())
        # return metric_result_str

        return f'mAP: {mAP}, loss: {eval_loss}', {'mAP': mAP, 'loss': eval_loss}


    def collate_batch(self, batch_list):
        # Have to overwrite this here
        batch_size = len(batch_list)
        # [{label: {}, all_results: {}}, {label: {}, all_results: {}}, ...]
        key_to_list = {}
        for key in batch_list[0].keys():
            key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]

        input_dict = {}
        n_models = len(self.dataset_cfg.MODELS)
        for key, val_list in key_to_list.items():
            if key == 'label':
                val_list_single = [x['mAP']['idx'] for x in val_list]
                val_list = [x['mAPs'] for x in val_list]
                val_list_single = torch.from_numpy(np.array(val_list_single)).to(torch.float32)
                val_list = torch.from_numpy(np.array(val_list)).to(torch.float32)
                input_dict[key] = val_list
                input_dict['label_single'] = val_list_single
            elif key == 'all_results':
                # key = label, val_list = [{}_1, {}_2, ...]
                inner_key_to_list = {}
                # val_list = batch size length list, each with n_models length list, of actual dict
                for key in val_list[0][0].keys():
                    inner_key_to_list[key] = [[val_list[bs_idx][model_idx][key] for model_idx in range(n_models)] for bs_idx in range(batch_size)]
                # Now inner_key_to_list['scenario_id'] = bs x n_models x *dims

                # keys = ['scenario_id', 'pred_trajs', 'pred_scores', 'object_id', 'object_type', 'gt_trajs', 'track_index_to_predict']
                for key, val_list in inner_key_to_list.items():
                    if key in ['scenario_id', 'object_id', 'object_type']:
                        input_dict[key] = np.array(val_list)[:, 0]
                    elif key in ['gt_trajs', 'object_id']:
                        input_dict[key] = torch.from_numpy(np.array(val_list)[:, 0])
                    elif key in ['pred_trajs', 'pred_scores']:
                        input_dict[key] = torch.from_numpy(np.array(val_list))
                    else:
                        # Ignore it, such as track_index_to_predict
                        pass
                        #raise ValueError('Unsupported inner key: ' + key)

        batch_sample_count = [x['all_results'][0]['track_index_to_predict'].size for x in batch_list]
        batch_dict = {'batch_size': batch_size, 'input_dict': input_dict, 'batch_sample_count': batch_sample_count}
        return batch_dict