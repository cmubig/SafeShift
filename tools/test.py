# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path
from easydict import EasyDict

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from mtr.datasets import build_dataloader
from mtr.models import model as model_utils
from mtr.models.vrnn import SimpleNN as VRNN
from mtr.models.motion_cnn import MotionCNN
from mtr.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, action='append', help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, action='append', help='checkpoint to start from')
    parser.add_argument('--save_train', action='store_true', help='whether or not to save train preds as well')
    parser.add_argument('--save_val', action='store_true', help='whether or not to save val preds as well')
    parser.add_argument('--no_test', action='store_true', help='whether or not to disable test preds as well')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--cache_only', action='store_true', help='whether or not to cache the create_scene_data results')
    parser.add_argument('--cache_scene_dir', type=str, default='../../monet_shared/shared/mtr_process', help='where to cache create_scene_data results')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--cpu', action='store_true', default=False, help='Use cpu for model')

    args = parser.parse_args()
    assert args.cache_only or (len(args.cfg_file) == len(args.ckpt)), 'Mismatch in cfg_files and ckpts'
    assert not args.cache_only or (args.save_train and args.save_val), 'Must save train and val if caching'
    if len(args.cfg_file) == 1:
        ensemble = False
        args.cfg_file = args.cfg_file[0]
        args.ckpt = args.ckpt[0] if not args.cache_only else None
    else:
        ensemble = True
    if ensemble:
        print('Not yet supported')
        import sys; sys.exit(0)

    cfg['CPU'] = args.cpu

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False, pickle_name='result.pkl'):
    # load checkpoint
    if args.ckpt is not None: 
        it, epoch = model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    else:
        it, epoch = -1, -1
    if not args.cpu:
        model.cuda()
    model.cuda()

    logger.info(f'*************** LOAD MODEL (epoch={epoch}, iter={it}) for EVALUATION *****************')
    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file, pickle_name=pickle_name
    )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False, pickle_name='result.pkl'):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_val.txt')
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / 'tensorboard_val'))
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        it, epoch = model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        logger.info(f'*************** LOAD MODEL (epoch={epoch}, iter={it}) for EVALUATION *****************')
        if not args.cpu:
            model.cuda()
        model.cuda()

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id)
        tb_dict = eval_utils.eval_one_epoch(
            cfg, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir, save_to_file=args.save_to_file, pickle_name=pickle_name
        )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar('eval/' + key, val, cur_epoch_id)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus
          
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all and not args.cache_only:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        if Path(args.ckpt).stem == 'best_model':
            epoch_id = 'best'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id)
    else:
        epoch_id = None
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    if not args.no_test:
        test_set, test_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            batch_size=args.batch_size,
            dist=dist_test, workers=args.workers, logger=logger, training=False
        )
    if args.save_train:
        train_set, train_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            batch_size=args.batch_size,
            dist=dist_test, workers=args.workers, logger=logger, training=False,
            override_training_for_output=True
        )
    if args.save_val:
        new_cfg = EasyDict(cfg.copy())
        new_cfg.DATA_CONFIG.SPLIT_DIR.test = new_cfg.DATA_CONFIG.SPLIT_DIR.val
        new_cfg.DATA_CONFIG.INFO_FILE.test = new_cfg.DATA_CONFIG.INFO_FILE.val
        val_set, val_loader, sampler = build_dataloader(
            dataset_cfg=new_cfg.DATA_CONFIG,
            batch_size=args.batch_size,
            dist=dist_test, workers=args.workers, logger=logger, training=False
        )
    
    if args.cache_only:
        assert 'mtr.datasets.waymo.waymo_dataset.WaymoDataset' in str(type(test_set))
        logger.info('Caching created scenes on HDD')
        logger.info('Caching train')
        train_set.cache_scenes(args.cache_scene_dir)
        logger.info('Caching val')
        val_set.cache_scenes(args.cache_scene_dir)
        logger.info('Caching test')
        test_set.cache_scenes(args.cache_scene_dir)
        import sys; sys.exit(0)

    if 'VRNN' in cfg.MODEL.keys():
        model = VRNN(config=cfg.MODEL)
    elif 'MotionCNN' in cfg.MODEL.keys():
        model = MotionCNN(config=cfg.MODEL)
    else:
        model = model_utils.MotionTransformer(config=cfg.MODEL)

    # TODO: Allow spilling into unified memory!
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 2
    # config.gpu_options.allow_growth = True
    # session = tf.compat.v1.InteractiveSession(config=config)
    # TODO: maybe add option unload the model in eval_one_epoch, before/after collecting all the predictions

    #import pdb; pdb.set_trace()
    with torch.no_grad():
        if args.eval_all:
            if args.save_train:
                repeat_eval_ckpt(model, train_loader, args, eval_output_dir, logger, 
                                ckpt_dir, dist_test=dist_test, pickle_name='result_train.pkl')
            if args.save_val:
                repeat_eval_ckpt(model, val_loader, args, eval_output_dir, logger, 
                                ckpt_dir, dist_test=dist_test, pickle_name='result_val.pkl')
            if not args.no_test:
                repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
        else:
            if args.save_train:
                eval_single_ckpt(model, train_loader, args, eval_output_dir, logger, epoch_id, 
                                dist_test=dist_test, pickle_name='result_train.pkl')
            if args.save_val:
                eval_single_ckpt(model, val_loader, args, eval_output_dir, logger, epoch_id, 
                                dist_test=dist_test, pickle_name='result_val.pkl')
            if not args.no_test:
                eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)


if __name__ == '__main__':
    main()
