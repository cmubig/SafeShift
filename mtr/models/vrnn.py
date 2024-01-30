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
import pandas as pd
import numpy as np
from torch.autograd import Variable
from mtr.utils import common_utils
from easydict import EasyDict

from mtr.models.utils import polyline_encoder
from mtr.config import cfg

class FutVRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        feat_enc = EasyDict(self.config.feat_enc_x)
        self.f_x = MLP(feat_enc)
        self.f_x_out_size = feat_enc.out_size
        self.dim = config.dim
        self.use_coll_loss = cfg.DATA_CONFIG.get('USE_COLL_LOSS', False)

        # TODO: Make encoder AND prior both conditioned upon history_features from PolylineEncoder

        enc = EasyDict(self.config.encoder)
        self.enc = MLP(enc)
        self.enc_out_size = enc.out_size
        assert self.enc_out_size % 2 == 0, \
            f"Encoder's output size must be divisible by 2"
        self.z_dim = int(self.enc_out_size / 2)

        # x - prior
        self.prior = MLP(EasyDict(self.config.prior))

        # x - feature 
        self.f_z = MLP(EasyDict(self.config.feat_enc_z))
        
        # x - decoder
        self.dec = MLP(EasyDict(self.config.decoder))

        # recurrent network 
        rnn = EasyDict(self.config.rnn)
        self.num_layers = rnn.num_layers
        self.rnn_dim = rnn.hidden_size
        self.rnn = nn.GRU(rnn.in_size, self.rnn_dim, self.num_layers)

        traj_score_config = EasyDict({
            'in_size': self.config.decoder.in_size,
            'hidden_size': [64, 32],
            'out_size': 1,
            'layer_norm': True,
            'dropout': 0.0
        })
        self.traj_score = MLP(traj_score_config)

    
    # Input = T x B x d -> Output T x B x B
    def simple_adjs(self, hist_abs, seq_start_end):
        num_batch = hist_abs.shape[1]
        hist_adj = torch.zeros((hist_abs.shape[0], num_batch, num_batch), device=hist_abs.device)
        for (start, end) in seq_start_end:
            hist_adj[:, start:end, start:end] = 1
        return hist_adj

    def simple_distsim_adjs(self, hist_abs, seq_start_end, sigma, seq_adj=None):
        if seq_adj is None:
            seq_adj = self.simple_adjs(hist_abs[0].unsqueeze(0), seq_start_end)[0]
        num_batch = hist_abs.shape[1]
        hist_adj = torch.zeros((hist_abs.shape[0], num_batch, num_batch), device=hist_abs.device)
        for t in range(hist_abs.shape[0]):
            abs_diffs = hist_abs[t].unsqueeze(-2) - hist_abs[t]
            dists = torch.sqrt(torch.sum(abs_diffs**2, dim=-1))
            # Mask out non-adjacent ones
            hist_adj[t] = torch.exp(-dists/sigma) * seq_adj 
        return hist_adj
    
    # TODO: decide on whether to decode 80 or 16 (i.e. sample interval of 1 or 5)
    def forward_train(self, hist_feats, fut, fut_mask, loss_weights=None):
        KLD = torch.zeros(1).to(fut.device)
        NLL = torch.zeros(1).to(fut.device)
        fut = fut[4::5]
        fut_mask = fut_mask[4::5]
        timesteps, num_agents, _ = fut.shape
        h = Variable(torch.zeros(
            self.num_layers, num_agents, self.rnn_dim)).to(fut.device)
        
        stop_idxs = fut_mask.to(torch.int).argmin(dim=0)
        stop_idxs[fut_mask.to(torch.int).min(dim=0)[0] == 1] = len(fut_mask)
        # Start with Pointnet encoded hist_feat
        h[-1] = hist_feats

        for t in range(0, timesteps):
            # x - extract features at step t
            x_t = fut[t]
            f_x_t = self.f_x(x_t) 

            # x - encode step t (encoder)
            x_enc_embedding = torch.cat([f_x_t,  h[-1]], 1)
            x_enc_t = self.enc(x_enc_embedding)
            x_enc_mean_t = x_enc_t[:, :self.z_dim]
            x_enc_logvar_t = x_enc_t[:, self.z_dim:]

            # x - encode step t (prior)
            prior_embedding = torch.cat([h[-1]], 1)
            x_prior_t = self.prior(prior_embedding)
            x_prior_mean_t = x_prior_t[:, :self.z_dim]
            x_prior_logvar_t = x_prior_t[:, self.z_dim:]

            # z - sample from latent space 
            z_t = self._reparameterize(x_enc_mean_t, x_enc_logvar_t)
            
            # z - extract feature at step t
            f_z_t = self.f_z(z_t)

            # z - decode step t to generate x_t
            x_dec_embedding = torch.cat([f_z_t, h[-1]], 1)
            x_dec_t = self.dec(x_dec_embedding)
            x_dec_mean_t = x_dec_t[:, :self.dim]
            x_dec_logvar_t = x_dec_t[:, self.dim:]

            # recurrence
            h_embedding = torch.cat([f_x_t, f_z_t], 1).unsqueeze(0)
            _, h = self.rnn(h_embedding, h)

            update_mask = t < stop_idxs
            if update_mask.any():
                KLD += self._kld(x_enc_mean_t[update_mask], x_enc_logvar_t[update_mask],
                                x_prior_mean_t[update_mask], x_prior_logvar_t[update_mask], 
                                loss_weights[update_mask])
                NLL += self._nll_gauss(x_dec_mean_t[update_mask], x_dec_logvar_t[update_mask], x_t[update_mask], 
                                       loss_weights[update_mask])
            if KLD.isnan() or NLL.isnan():
                breakpoint()

        return KLD, NLL
    
    def forward_predict(self, hist_feats, timesteps=16):
    # def forward_predict(self, hist_feats, timesteps=80):
        num_agents, _ = hist_feats.shape

        h = Variable(torch.zeros(
            self.num_layers, num_agents, self.rnn_dim)).to(hist_feats.device)
        h[-1] = hist_feats

        samples = torch.zeros(timesteps, num_agents, self.dim).to(h.device)

        score = 0
        for t in range(timesteps):
            # x - encode hidden state to generate latent space (prior)
            prior_embedding = torch.cat([h[-1]], 1)
            x_prior_t = self.prior(prior_embedding)
            x_prior_mean_t = x_prior_t[:, :self.z_dim]
            x_prior_logvar_t = x_prior_t[:, self.z_dim:]

            # z - sample from latent space 
            z_t = self._reparameterize(x_prior_mean_t, x_prior_logvar_t)
            
            # z - extract feature at step t
            f_z_t = self.f_z(z_t)

            # z - decode step t to generate x_t
            x_dec_embedding = torch.cat([f_z_t, h[-1]], 1)
            x_dec_t = self.dec(x_dec_embedding)
            x_dec_mean_t = x_dec_t[:, :self.dim]
            step_score = self.traj_score(x_dec_embedding)
            score += step_score
            
            # (N, D)
            # if self.use_coll_loss:
            #     if not self.training:
            #         samples[t] = x_dec_mean_t.data
            #     else:
            #         # preserve gradient
            #         samples[t] = x_dec_mean_t
            # else:
            #     samples[t] = x_dec_mean_t.data
            samples[t] = x_dec_mean_t.data

            # x - extract features from decoded latent space (~ 'x')
            f_x_t = self.f_x(x_dec_mean_t)

            # recurrence
            h_embedding = torch.cat([f_x_t, f_z_t], 1).unsqueeze(0)
            _, h = self.rnn(h_embedding, h)

        return samples, score / timesteps

    def _reparameterize(
        self, mean: torch.tensor, log_var: torch.tensor
    ) -> torch.tensor:
        """ Generates a sample z for the decoder using the mean, logvar parameters
        outputed by the encoder (during training) or prior (during inference). 
            z = mean + sigma * eps
        See: https://www.tensorflow.org/tutorials/generative/cvae
        
        Inputs:
        -------
        mean[torch.tensor]: mean of a Gaussian distribution 
        log_var[torch.tensor]: standard deviation of a Gaussian distribution.
                
        Outputs:
        --------
        z[torch.tensor]: sampled latent value. 
        """
        logvar = torch.exp(log_var * 0.5).to(mean.device)
        # eps is a random noise
        eps = torch.rand_like(logvar).to(mean.device)
        return eps.mul(logvar).add(mean)

    def _kld(
        self, mean_enc: torch.tensor, logvar_enc: torch.tensor, 
        mean_prior: torch.tensor, logvar_prior: torch.tensor, loss_weights: torch.tensor
    ) -> torch.tensor:
        """ KL Divergence between the encoder and prior distributions:
            x1 = log(sigma_p / sigma_e)
            x2 = sigma_m ** 2 / sigma_p ** 2
            x3 = (mean_p - mean_e) ** 2 / sigma_p ** 2
            KL(p, q) = 0.5 * (x1 + x2 + x3 - 1)
        See: https://stats.stackexchange.com/questions/7440/ \
                kl-divergence-between-two-univariate-gaussians
        
        Inputs:
        -------
        mean_enc[torch.tensor]: encoder's mean at time t. 
        logvar_enc[torch.tensor]: encoder's variance at time t.
        mean_prior[torch.tensor]: prior's mean at time t. 
        logvar_prior[torch.tensor]: prior's variance at time t.
        
        Outputs:
        --------
        kld[torch.tensor]: Kullback-Leibler divergence between the prior and
        encoder's distributions time t. 
        """
        x1 = torch.sum((logvar_prior - logvar_enc), dim=1)
        x2 = torch.sum(torch.exp(logvar_enc - logvar_prior), dim=1)
        x3 = torch.sum((mean_enc - mean_prior).pow(2) /
                       (torch.exp(logvar_prior)), dim=1)
        kld_element = (x1 - mean_enc.size(1) + x2 + x3) * loss_weights
        return torch.mean(0.5 * kld_element)

    def _nll_gauss(
        self, mean: torch.tensor, logvar: torch.tensor, x: torch.tensor, loss_weights: torch.tensor
    ) -> torch.tensor:
        """ Negative Log-Likelihood with Gaussian.
            x1 = (x - mean) ** 2 / var
            x2 = logvar 
            x3 = const = 1 + log(2*pi)
            nll = 0.5 * (x1 + x2 + x3)
        See: https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
        
        Inputs:
        -------
        mean[torch.tensor]: decoder's mean at time t.
        logvar[torch.tensor]: decoder's variance a time t.
        x[torch.tensor]: ground truth X at time t.
        
        Outpus:
        -------
        nll[torch.tensor]: Gaussian Negative Log-Likelihood at time t. 
        """
        x1 = torch.sum(((x - mean).pow(2)) / torch.exp(logvar), dim=1)
        x2 = x.size(1) * np.log(2 * np.pi)
        x3 = torch.sum(logvar, dim=1)
        nll = torch.mean((0.5 * (x1 + x2 + x3)) * loss_weights)
        return nll

class SimpleNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        config = config.VRNN
        self.config = config
        self.cpu = cfg.get('CPU', False)

        # Feature extractor
        hist_feats = 96
        self.agent_feats = self.build_polyline_encoder(config.dim + 1, hidden_dim = 96, num_layers = 3,
                                                         out_channels=hist_feats, gru=True) 
        self.other_feats = self.build_polyline_encoder(config.dim + 1, hidden_dim = 96, num_layers = 3,
                                                         out_channels=hist_feats, gru=True) 
        
        # config.encoder.in_size += hist_feats
        # config.prior.in_size += hist_feats
        # config.decoder.in_size += hist_feats
        # config.rnn.in_size += hist_feats
        if config.dim == 4:
            config.feat_enc_x.in_size = 4
            config.decoder.out_size = 8
        self.fut_vrnn = FutVRNN(config)

        # Simple MLP decoder for now, GRU later?
        # traj_pred_config = EasyDict({
        #     'in_size': hist_feats,
        #     'hidden_size': [64, 32],
        #     'out_size': 160,
        #     'layer_norm': True,
        #     'dropout': 0.0
        # })
        # self.traj_pred = MLP(traj_pred_config)

        # def avg_pool(features, adj):
        #     # Shape = B x B x d
        #     weighted = (features.unsqueeze(0).permute(2, 0, 1)*adj).permute(1, 2, 0)
        #     # Shape = B x d
        #     weighted = weighted.sum(dim=-2)/adj.sum(dim=-1).unsqueeze(-1)
        #     return weighted
        # self.graph = avg_pool

        # lg_config = EasyDict({
        #     'in_size': hist_feats + hist_feats,
        #     'hidden_size': [64],
        #     'out_size': hist_feats,
        #     'layer_norm': True,
        #     'dropout': 0.0
        # })
        # self.lg = MLP(lg_config)
        self.fe_weight_input = cfg.DATA_CONFIG.get('FE_WEIGHT_INPUT', False) 
        if self.fe_weight_input:
            fe_weight_config = EasyDict({
                'in_size': 1,
                'hidden_size': [4],
                'out_size': 16,
                'layer_norm': True,
                'dropout': 0.0
            })
            self.fe_weight = MLP(fe_weight_config)
            lg_in = hist_feats + hist_feats + 16
        else:
            lg_in = hist_feats + hist_feats
        self.lg = nn.Linear(lg_in, hist_feats)
        self.use_coll_loss = cfg.DATA_CONFIG.get('USE_COLL_LOSS', False)


    def forward(self, batch_dict):
        input_dict = batch_dict['input_dict']
        if not self.cpu:
            obj_trajs, obj_trajs_mask = input_dict['obj_trajs'].cuda(), input_dict['obj_trajs_mask'].cuda() 
            obj_trajs_fut, obj_trajs_fut_mask = input_dict['obj_trajs_future_state'].cuda(), input_dict['obj_trajs_future_mask'].cuda() 
        else:
            obj_trajs, obj_trajs_mask = input_dict['obj_trajs'], input_dict['obj_trajs_mask'] 
            obj_trajs_fut, obj_trajs_fut_mask = input_dict['obj_trajs_future_state'], input_dict['obj_trajs_future_mask'] 
        track_index_to_predict = input_dict['track_index_to_predict']
        assert obj_trajs_mask.dtype == torch.bool

        center_tracks = torch.stack([obj_trajs[center_idx, track_idx] for center_idx, track_idx in \
                                     enumerate(track_index_to_predict)])
        center_mask = torch.stack([obj_trajs_mask[center_idx, track_idx] for center_idx, track_idx in \
                                   enumerate(track_index_to_predict)])

        center_tracks_fut = torch.stack([obj_trajs_fut[center_idx, track_idx] for center_idx, track_idx in \
                                     enumerate(track_index_to_predict)])
        center_mask_fut = torch.stack([obj_trajs_fut_mask[center_idx, track_idx] for center_idx, track_idx in \
                                   enumerate(track_index_to_predict)])

        # TODO: use xy and also sd potentially, if config is set to "both"
        if self.config.dim == 2:
            assert center_tracks.shape[-1] == 29
            agent_in = torch.cat([center_tracks[..., :2], center_mask.unsqueeze(-1)], dim=-1)
        elif self.config.dim == 4:
            assert center_tracks.shape[-1] == 31
            agent_in = torch.cat([center_tracks[..., :2], center_tracks[..., -2:], center_mask.unsqueeze(-1)], dim=-1)

        features = self.agent_feats(agent_in.unsqueeze(1), center_mask.unsqueeze(1)).squeeze(1)

        # TODO: encode other hist features as well, in agent-centric
        # Then agent-by-agent refine it, weighted by closest distance in hist reached
        # Effectively a distance-weighted avg pooling, as used in T2FPV
        if self.config.dim == 2:
            other_in = torch.cat([obj_trajs[..., :2], obj_trajs_mask.unsqueeze(-1)], dim=-1)
        elif self.config.dim == 4:
            other_in = torch.cat([obj_trajs[..., :2], obj_trajs[..., -2:], obj_trajs_mask.unsqueeze(-1)], dim=-1)

        other_feat = self.other_feats(other_in, obj_trajs_mask)

        dists = torch.linalg.norm(agent_in.unsqueeze(1)[..., :2] - other_in[..., :2], axis=-1)
        shared_mask = obj_trajs_mask & center_mask.unsqueeze(1)
        dists[~shared_mask] = torch.inf
        min_dists = dists.min(dim=-1)[0]
        adjs = torch.exp(-min_dists/1.2) 
        weighted_other = other_feat*adjs.unsqueeze(-1)
        weighted_other = weighted_other.sum(dim=1)/adjs.sum(dim=-1).unsqueeze(-1)

        combined = torch.cat([features, weighted_other], dim=-1)
        if self.fe_weight_input:
            fe_inputs = self.fe_weight(input_dict['fe_scores'].to(combined.device).unsqueeze(-1))
            combined = torch.cat([combined, fe_inputs], dim=-1)
        features = self.lg(combined)

        # TODO: if dim is 4, as in SD, adjust accordingly
        if self.config.dim == 2:
            fut_xy = center_tracks_fut[:, :, :2].permute((1, 0, 2))
        elif self.config.dim == 4:
            fut_xy = torch.cat([center_tracks_fut[:, :, :2], center_tracks_fut[:, :, -2:]], dim=-1).permute((1, 0, 2))
        fut_mask = center_mask_fut.to(torch.bool).permute((1, 0))

        if 'scores' in input_dict:
            loss_weights = input_dict['scores'].to(features.device)
        else:
            loss_weights = torch.ones((features.shape[0],)).to(features.device)
        kld, nll = self.fut_vrnn.forward_train(features, fut_xy, fut_mask, loss_weights=loss_weights)

        sample, score = self.fut_vrnn.forward_predict(features.repeat(6, 1))

        # TODO: interpolate to 80 timesteps
        # Current shape: 16 x (6*B) x 2
        # Initial is always zero
        prepend = torch.zeros((1, *sample.shape[1:])).to(sample.device)
        with_cur = torch.cat([prepend, sample], dim=0)
        interpolated = torch.nn.functional.interpolate(with_cur.permute((1, 2, 0)), 85, mode='linear', align_corners=False)
        interpolated = interpolated[:, :, 3:-2]
        interpolated = interpolated.permute((2, 0, 1))

        #samples = sample.reshape(16, 6, -1, 2)
        samples = interpolated.reshape(80, 6, -1, self.config.dim)
        scores = score.reshape(6, -1, 1)

        # B x n_samples x T x 2
        pred_trajs = samples.permute((2, 1, 0, 3))
        pred_trajs = pred_trajs[..., :2]

        # B x n_samples
        pred_scores = scores.squeeze(-1).permute((1, 0))

        if self.training:
            final_idxs = batch_dict['input_dict']['center_gt_final_valid_idx'].to(torch.long).to(features.device)
            

            updated_center_fut = center_tracks_fut
            fdes = torch.stack([pred_trajs[i, :, final_idx] - updated_center_fut[i, final_idx, :2].unsqueeze(0) for i, final_idx in enumerate(final_idxs)])
            fdes = torch.linalg.norm(fdes, dim=-1)

            # Collision loss
            # See obj_trajs_fut and obj_trajs_fut_mask
            # Shape: B x n_agents_per_center_object x 6 x 80 x 2
            if self.use_coll_loss:
                collision_dists = pred_trajs.unsqueeze(1) - obj_trajs_fut.unsqueeze(2)[..., :2]
                # Shape: B x n_agents_per_center_object x 6 x 80
                collision_dists = torch.linalg.norm(collision_dists, dim=-1)
                collision_dists = collision_dists.permute(0, 1, 3, 2)

                coll_mask = (collision_dists < 1)
                # coll_mask = (collision_dists < 100000)

                coll_mask[~obj_trajs_fut_mask.to(torch.bool)] = False
                # Mask out self-collisions
                for i, track_idx in enumerate(track_index_to_predict):
                    coll_mask[i, track_idx] = False
                # Shape: B x n_agents_per_center_object x 80 x 6; take "any" over time
                # coll_any_mask = coll_mask.any(dim=-2)
                # collision_dists = collision_dists.permute(0, 1, 3, 2)
                # coll_loss = torch.exp(-collision_dists)
                # coll_loss = coll_loss.sum(dim=1)
                # coll_loss = coll_loss[coll_any_mask].sum() / (coll_loss.shape[0]*6)

                # Just count how close you are at 0.25m, divide by B for relative importance to other losses
                # Kinda similar to SafeCritic, where least loss/most reward is fewest collisions
                # Loses gradient here, but it's okay for now, getting absorbed into score_loss
                collision_dists[~coll_mask] = torch.inf
                # coll_loss = torch.exp(-collision_dists)
                coll_loss = torch.exp(-collision_dists/0.25)

                # Sum over time and over other agents
                coll_loss = coll_loss.sum(dim=(1, 2))
                # TODO: Could also consider using coll_loss to influence score; where "dirty" paths cannot have the label?
                # Like just add to FDE in some sort of balancing act...
                # Can also consider predicting *more* than 6 future trajectories to really make this possible, or
                # do multiple iterations (at least during training)?
                
                # Could also try to potentially have a refining phase, where you fix the endpoint, but then try to minimize 
                # collisions along the way...seems like classical planning tbh but idk, OR fix the end point, then condition on it
                # and generate 6 possible trajectories for each, and select the least colliding one during train time?
                # So like
                #  1. generate 6 possible trajs
                #  2. select best traj by fde
                #  3. refine it somehow?
                fdes = coll_loss
                # fdes = fdes + 100*coll_loss

                # coll_loss = coll_loss[obj_trajs_fut_mask.to(torch.bool)].mean()*80
                #coll_loss = coll_loss[obj_trajs_fut_mask.to(torch.bool)].mean()
            # Best predicted traj
            labels = fdes.argmin(dim=-1)
            # scalar
            score_loss = torch.nn.functional.cross_entropy(pred_scores, labels)

            # 3 loss components: nll_loss, kld_loss, score_loss, pred_loss
            # if self.use_coll_loss:
            #     loss = kld + nll + score_loss + coll_loss
            # else:
            loss = kld + nll + score_loss

            # if self.use_coll_loss:
            #     tb_dict = {'loss': loss.item(), 'nll_loss': nll.item(), 'kld_loss': kld.item(), 
            #                'score_loss': score_loss.item(), 'coll_loss': coll_loss.item()}
            #     disp_dict = {'loss': loss.item(), 'nll_loss': nll.item(), 'kld_loss': kld.item(), 
            #                'score_loss': score_loss.item(), 'coll_loss': coll_loss.item()}
            # else:
            tb_dict = {'loss': loss.item(), 'nll_loss': nll.item(), 'kld_loss': kld.item(), 
                    'score_loss': score_loss.item()}
            disp_dict = {'loss': loss.item(), 'nll_loss': nll.item(), 'kld_loss': kld.item(), 
                    'score_loss': score_loss.item()}
            return loss, tb_dict, disp_dict
        
        # pred_trajs = best_traj.unsqueeze(1).repeat((1, 6, 1, 1)).contiguous()
        # pred_scores = torch.ones(pred_trajs.shape[:2], dtype=pred_trajs.dtype, device=pred_trajs.device) * 1/6
        batch_dict['pred_scores'] = pred_scores.contiguous()
        batch_dict['pred_trajs'] = pred_trajs.contiguous()
        return batch_dict

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
    