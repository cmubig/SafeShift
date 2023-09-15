# Motion CNN
# Stepan Konev


import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .context_encoder import build_context_encoder
from .motion_decoder import build_motion_decoder
from .utils.common_layers import build_mlps, MLP
import pandas as pd
import numpy as np
from torch.autograd import Variable
from mtr.utils import common_utils
from easydict import EasyDict
import timm

from abc import ABC

from mtr.config import cfg

# TODO: Implement weight loss, fe_weight_input, & coll_loss...

class Loss(ABC, nn.Module):
    def _precision_matrix(shape, sigma_xx, sigma_yy):
        assert sigma_xx.shape[-1] == 1
        assert sigma_xx.shape == sigma_yy.shape
        batch_size, n_modes, n_future_timstamps = \
            sigma_xx.shape[0], sigma_xx.shape[1], sigma_xx.shape[2]
        sigma_xx_inv = 1 / sigma_xx
        sigma_yy_inv = 1 / sigma_yy
        return torch.cat(
            [sigma_xx_inv, torch.zeros_like(sigma_xx_inv),
            torch.zeros_like(sigma_yy_inv), sigma_yy_inv], dim=-1) \
            .reshape(batch_size, n_modes, n_future_timstamps, 2, 2)

    def _log_N_conf(self, data_dict, prediction_dict, frenet=False, use_coll_loss=False):
        gt = data_dict['future_local'].unsqueeze(1)
        diff = (prediction_dict['xy'] - gt) * \
            data_dict['future_valid'][:, None, :, None]
        assert torch.isfinite(diff).all()
        precision_matrices = self._precision_matrix(
            prediction_dict['sigma_xx'], prediction_dict['sigma_yy'])
        assert torch.isfinite(precision_matrices).all()

        log_confidences = torch.log_softmax(
            prediction_dict['confidences'], dim=-1)

        if use_coll_loss:
            pred_trajs = prediction_dict['xy']
            # Need ground truth of *other* agents, not just self, as well as valid
            obj_trajs_fut, obj_trajs_fut_mask = data_dict['obj_trajs_future_state'], data_dict['obj_trajs_future_mask'] 
            collision_dists = pred_trajs.unsqueeze(1) - obj_trajs_fut.unsqueeze(2)[..., :2]
            # Shape: B x n_agents_per_center_object x 6 x 80
            collision_dists = torch.linalg.norm(collision_dists, dim=-1)
            collision_dists = collision_dists.permute(0, 1, 3, 2)

            coll_mask = (collision_dists < 1)

            coll_mask[~obj_trajs_fut_mask.to(torch.bool)] = False
            # Mask out self-collisions
            track_index_to_predict = data_dict['track_index_to_predict']
            for i, track_idx in enumerate(track_index_to_predict):
                coll_mask[i, track_idx] = False

            collision_dists[~coll_mask] = torch.inf
            # coll_loss = torch.exp(-collision_dists)
            coll_loss = torch.exp(-collision_dists/0.25)

            # Sum over time and over other agents
            coll_loss = coll_loss.sum(dim=(1, 2))

            labels = coll_loss.argmin(dim=-1)
            # scalar
            score_loss = torch.nn.functional.cross_entropy(prediction_dict['confidences'], labels)

        assert torch.isfinite(log_confidences).all()
        bilinear = diff.unsqueeze(-2) @ precision_matrices @ diff.unsqueeze(-1)
        bilinear = bilinear[:, :, :, 0, 0]
        assert torch.isfinite(bilinear).all()

        # TODO: use data_dict['scores'] to weight losses
        bilinear = bilinear * data_dict['scores'][:, None, None]

        log_N = -0.5 * np.log(2 * np.pi) - 0.5 * torch.log(
            prediction_dict['sigma_xx'] * prediction_dict['sigma_yy']
            ).squeeze(-1) - 0.5 * bilinear

        # if Frenet+ strategy, incorporate sd coordinates here?
        if frenet:
            gt_sd = data_dict['sd_trajs_fut'].unsqueeze(1)
            diff_sd = (prediction_dict['sd'] - gt_sd) * \
                data_dict['future_valid'][:, None, :, None]
            precision_matrices_sd = self._precision_matrix(
                prediction_dict['sigma_ss'], prediction_dict['sigma_dd']
            )
            bilinear_sd = diff_sd.unsqueeze(-2) @ precision_matrices_sd @ diff_sd.unsqueeze(-1) 
            bilinear_sd = bilinear_sd[:, :, :, 0, 0]
            log_N_sd = -0.5 * np.log(2 * np.pi) - 0.5 * torch.log(
                prediction_dict['sigma_ss'] * prediction_dict['sigma_dd']
                ).squeeze(-1) - 0.5 * bilinear_sd
            log_N = log_N + log_N_sd
        if use_coll_loss:
            return log_N, log_confidences, score_loss
        else:
            return log_N, log_confidences


class NLLGaussian2d(Loss):
    def __init__(self, use_coll_loss=False):
        super().__init__()
        self.use_coll_loss = use_coll_loss

    def forward(self, data_dict, prediction_dict, frenet=False):
        if self.use_coll_loss:
            log_N, log_confidences, coll_loss = self._log_N_conf(data_dict, prediction_dict, frenet=frenet, use_coll_loss=self.use_coll_loss)
        else:
            log_N, log_confidences = self._log_N_conf(data_dict, prediction_dict, frenet=frenet, use_coll_loss=self.use_coll_loss)

        # Augment log_confidences with coll_loss effectively
        if self.use_coll_loss:
            assert torch.isfinite(log_N).all()
            log_L = torch.logsumexp(log_N.sum(dim=2) + log_confidences, dim=1)
            #log_L = torch.logsumexp(log_N.sum(dim=2), dim=1)
            assert torch.isfinite(log_L).all()
            # TODO: try just cl, without wl or fe?
            return 100*coll_loss - log_L.mean()
            #return coll_loss - log_L.mean()
        else:
            assert torch.isfinite(log_N).all()
            log_L = torch.logsumexp(log_N.sum(dim=2) + log_confidences, dim=1)
            assert torch.isfinite(log_L).all()
            return -log_L.mean()

class MotionCNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        config = config.MotionCNN
        self.config = config
        self.cpu = cfg.get('CPU', False)

        self.use_coll_loss = cfg.DATA_CONFIG.get('USE_COLL_LOSS', False)
        self.loss_module = NLLGaussian2d(self.use_coll_loss)

        # Feature extractor
        # TODO: for Frenet+ strategy, predict sd coordinates in addition to xy?
        # xy vs. xy sd
        self.frenet = cfg.DATA_CONFIG.get('APPEND_BOTH', False) or cfg.DATA_CONFIG.get('USE_FRENET', False)

        self.pred_dim = 2 if not self.frenet else 4
        # pred +  (sigma_xx, sigma_yy, visibility)
        self.total_dim = self.pred_dim*2 + 1

        self.output_dim = 6*self.total_dim*80 + 6
        model_name = 'resnet34'
        # model_name = 'efficientnet_lite0'
        self.resnet = timm.create_model(model_name=model_name, pretrained=True, in_chans=27, num_classes=self.output_dim)

        # TODO: for fe_score, modify self.resnet.fc to take in 512 + 16 shape instead
        self.fe_weight_input = cfg.DATA_CONFIG.get('FE_WEIGHT_INPUT', False) 
        if self.fe_weight_input:
            #hist_feats = self.resnet.fc.in_features
            hist_feats = [x for x in self.resnet.children()][-1].in_features
            fe_weight_config = EasyDict({
                'in_size': 1,
                'hidden_size': [4],
                'out_size': 16,
                'layer_norm': True,
                'dropout': 0.0
            })
            self.fe_weight = MLP(fe_weight_config)
            lg_in = hist_feats + 16
            self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
            # Needs fully connected layer in between
            self.resnet_fc = torch.nn.Linear(lg_in, self.output_dim)
        

    def forward(self, batch_dict):
        input_dict = batch_dict['input_dict']

        self.dict_to_cuda(input_dict)
        image_in = input_dict['raster']
        output = self.resnet(image_in)
        # - Perhaps just MLP encode the above, and then *refine* as we have been doing?
        # For SD, use input_dict['sd_trajs'] along with input_dict['track_index_to_predict] to get future SD
        if self.fe_weight_input:
            fe_inputs = self.fe_weight(input_dict['fe_scores'].to(output.device).unsqueeze(-1))
            combined = torch.cat([output, fe_inputs], dim=-1)
            output = self.resnet_fc(combined)

        prediction_dict = self.postprocess_predictions(output)
        loss = self.loss_module(input_dict, prediction_dict, self.frenet)

        if self.training:
            tb_dict = {'loss': loss.item()}
            disp_dict = {'loss': loss.item()}
            return loss, tb_dict, disp_dict
        
        # TODO: perhaps here add in the collision GT stuff?
        pred_scores = prediction_dict['confidences']
        pred_trajs = prediction_dict['xy']

        batch_dict['pred_scores'] = pred_scores
        batch_dict['pred_trajs'] = pred_trajs
        return batch_dict

    def dict_to_cuda(self, data_dict):
        gpu_required_keys = ['raster', 'future_valid', 'future_local', 'obj_trajs_future_state', 'obj_trajs_future_mask', 'scores']
        for key in gpu_required_keys:
            if not self.cpu:
                data_dict[key] = data_dict[key].float().cuda()
            else:
                data_dict[key] = data_dict[key].float()
    
    def limited_softplus(self, x):
        return torch.clamp(F.softplus(x), min=0.1, max=10)
    
    def postprocess_predictions(self, predicted_tensor):
        n_modes = 6
        n_timestamps = 80
        confidences = predicted_tensor[:, :n_modes]
        components = predicted_tensor[:, n_modes:]
        components = components.reshape(
            -1, n_modes, n_timestamps, self.total_dim)
        sigma_xx = components[:, :, :, 2:3]
        sigma_yy = components[:, :, :, 3:4]
        visibility = components[:, :, :, 4:5]
        ret =  {
            'confidences': confidences,
            'xy': components[:, :, :, :2],
            'sigma_xx': self.limited_softplus(sigma_xx),
            'sigma_yy': self.limited_softplus(sigma_yy),
            'visibility': visibility}
        if self.frenet:
            sd = components[:, :, :, 5:7]
            sigma_ss = components[:, :, :, 7:8]
            sigma_dd = components[:, :, :, 8:9]
            ret['sd'] = sd
            ret['sigma_ss'] = self.limited_softplus(sigma_ss)
            ret['sigma_dd'] = self.limited_softplus(sigma_dd)
        return ret

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                        % (filename, 'CPU' if to_cpu else 'GPU'))
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        # logger.info('==> Done')
        logger.info('==> Done (loaded %d/%d)' % (len(checkpoint['model_state']), len(checkpoint['model_state'])))

        return it, epoch

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        logger.info(f'The number of disk ckpt keys: {len(model_state_disk)}')
        model_state = self.state_dict()
        model_state_disk_filter = {}
        for key, val in model_state_disk.items():
            if key in model_state and model_state_disk[key].shape == model_state[key].shape:
                model_state_disk_filter[key] = val
            else:
                if key not in model_state:
                    print(f'Ignore key in disk (not found in model): {key}, shape={val.shape}')
                else:
                    print(f'Ignore key in disk (shape does not match): {key}, load_shape={val.shape}, model_shape={model_state[key].shape}')

        model_state_disk = model_state_disk_filter

        missing_keys, unexpected_keys = self.load_state_dict(model_state_disk, strict=False)

        logger.info(f'Missing keys: {missing_keys}')
        logger.info(f'The number of missing keys: {len(missing_keys)}')
        logger.info(f'The number of unexpected keys: {len(unexpected_keys)}')
        logger.info('==> Done (total keys %d)' % (len(model_state)))

        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        return it, epoch