# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .context_encoder import build_context_encoder
from .motion_decoder import build_motion_decoder
from .utils.common_layers import build_mlps, MLP

from mtr.models.utils import polyline_encoder

class MetaModel(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()
        
        loss_type = config.LOSS
        assert loss_type in ['cross_entropy', 'mse', 'map'], 'Unsupported loss type'
        self.loss_type = loss_type

        config = config.BASE
        self.config = config
        self.device = device

        self.agent_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.config.feat_enc_x.in_size + 1,
            hidden_dim=self.config.feat_enc_x.hidden_size,
            num_layers=self.config.feat_enc_x.n_layers,
            out_channels=self.config.feat_enc_x.out_size,
            gru=True if 'gru' in self.config.feat_enc_x and self.config.feat_enc_x.gru else False
        )

        self.future_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.config.feat_enc_x.in_size + 1,
            hidden_dim=self.config.feat_enc_x.hidden_size,
            num_layers=self.config.feat_enc_x.n_layers,
            out_channels=self.config.feat_enc_x.out_size,
            gru=True if 'gru' in self.config.feat_enc_x and self.config.feat_enc_x.gru else False
        )

        self.model_fut = MLP(config.feat_enc_model_fut, device=self.device)
        self.decoder = MLP(config.decoder, device=self.device)

        if self.loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif self.loss_type == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
    
    def get_loss(self, decoded, labels):
        if self.loss_type == 'mse':
            return self.criterion(decoded, labels)
        elif self.loss_type == 'cross_entropy':
            best_indices = torch.zeros_like(labels)
            best_indices[labels == labels.max(dim=1)[0].unsqueeze(-1)] = 1
            target = best_indices/best_indices.sum(dim=1).unsqueeze(-1)
            return self.criterion(decoded, target)
        elif self.loss_type == 'map':
            # best_mAP = labels.max(dim=1)[0]
            # missing_mAP = best_mAP.unsqueeze(-1) - labels
            # missing_mAP = (nn.functional.softmax(decoded, dim=1)*missing_mAP).sum(dim=1)
            # missing_mAP = missing_mAP*(best_mAP + 0.01)

            #mAP = (nn.functional.softmax(decoded, dim=1)*labels).sum(dim=1).mean()

            best_indices = torch.zeros_like(labels)
            best_indices[labels == labels.max(dim=1)[0].unsqueeze(-1)] = 1
            # Binary loss?
            acc = (best_indices*nn.functional.softmax(decoded, dim=1)).sum(dim=1).mean()
            return 1 - acc
        else:
            raise NotImplementedError('Other losses not yet implemented')
    
    def build_polyline_encoder(self, in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None, gru=False):
        if not gru:
            ret_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_pre_layers=num_pre_layers,
                out_channels=out_channels
            )
        else:
            ret_polyline_encoder = polyline_encoder.PointNetPolylineGRUEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_pre_layers=num_pre_layers,
                out_channels=out_channels
            )
        return ret_polyline_encoder

    def forward(self, batch_dict):
        labels = batch_dict['input_dict']['label'].cuda()
        labels_single = batch_dict['input_dict']['label_single'].cuda()
        gt_trajs = batch_dict['input_dict']['gt_trajs'].cuda()
        pred_trajs = batch_dict['input_dict']['pred_trajs'].cuda()
        pred_scores = batch_dict['input_dict']['pred_scores'].cuda()

        # Make agent centered
        gt_trajs_rel = torch.zeros_like(gt_trajs)
        gt_trajs_rel[:, 1:, :2] = gt_trajs[:, 1:, :2] - gt_trajs[:, :-1, :2]

        pred_trajs_rel = torch.zeros_like(pred_trajs)
        pred_trajs_rel[..., 1:, :2] = pred_trajs[..., 1:, :2] - pred_trajs[..., :-1, :2]
        pred_trajs_rel[..., 0, :2] = pred_trajs[..., 0, :2] - gt_trajs.unsqueeze(1).unsqueeze(1)[..., 10, :2]
        pred_scores = pred_scores.broadcast_to((pred_trajs_rel.shape[-2], *pred_scores.shape)).permute(1, 2, 3, 0).unsqueeze(-1)

        gt_trajs_valid = gt_trajs[:, :, -1].to(torch.bool)
        hist_gt = gt_trajs_rel[:, :11, :2]
        hist_valid = gt_trajs_valid[:, :11]

        hist_gt_in = torch.cat([hist_gt, hist_valid.unsqueeze(-1).to(torch.float32)], dim=-1).unsqueeze(1)
        hist_valid_in = hist_valid.unsqueeze(1)
        # polylines (batch_size, num_polylines, num_points_each_polylines, C):
        # polylines_mask (batch_size, num_polylines, num_points_each_polylines):
        encoded_hist = self.agent_polyline_encoder(hist_gt_in, hist_valid_in)

        ensembled_list = []
        for model_idx in range(pred_trajs_rel.shape[1]):
            pred_trajs_in = torch.cat([pred_trajs_rel[:, model_idx], pred_scores[:, model_idx]], dim=-1)
            pred_trajs_valid = torch.ones((pred_trajs_in.shape[:-1])).to(torch.bool).to(pred_trajs_in.device)
            encoded_preds = self.future_polyline_encoder(pred_trajs_in, pred_trajs_valid)

            # Max pool along the 6 candidate trajs
            #encoded_preds = encoded_preds.max(dim=-2)[0]
            # B x 96 -> B x 16
            #ensembled_list.append(self.model_fut(encoded_preds, self.training))

            # individual = encoded_preds.max(dim=-2)[0]
            individual = self.model_fut(encoded_preds, self.training).flatten(start_dim=1)
            one_hot = torch.nn.functional.one_hot(model_idx*torch.ones((individual.shape[0],), 
                                                  device=individual.device).to(torch.long), pred_trajs_rel.shape[1])
            ensembled_list.append(torch.cat([individual, one_hot], dim=-1))
        # decoder_in = torch.cat([encoded_hist.squeeze(1), *ensembled_list], dim=-1)
        # decoded = self.decoder(decoder_in, self.training)

        ensembled = torch.stack(ensembled_list, dim=1)
        decoder_in = torch.cat([encoded_hist.permute(0, 2, 1).broadcast_to(\
            (encoded_hist.shape[0], encoded_hist.shape[2], ensembled.shape[1])), 
            ensembled.permute(0, 2, 1)], dim=1).permute(0, 2, 1)
        decoded = self.decoder(decoder_in, self.training).squeeze(-1)

        if self.training:
            loss = self.get_loss(decoded, labels)
            tb_dict = {'loss': loss.item()}
            disp_dict = {'loss': loss.item()}
            return loss, tb_dict, disp_dict

        batch_dict['pred_labels'] = decoded
        return batch_dict

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


