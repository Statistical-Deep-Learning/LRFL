# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
import io
from pathlib import Path
import scipy

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import timm

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
# from util.mixup_multi_label import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy
from util.multi_label_loss import SoftTargetBinaryCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset, build_dataset_chest_xray
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import torch.nn.functional as F
import models_vit
import models_densenet
import models_resnet

from engine_finetune import train_one_epoch, evaluate_chestxray, evaluate_chestxray_before
from util.sampler import RASampler
# from apex.optimizers import FusedAdams
from libauc import losses
from torchvision import models
import timm.optim.optim_factory as optim_factory
from collections import OrderedDict
import warnings
import logging
from timm.utils import setup_default_logging
_logger = logging.getLogger('train')
_logger.setLevel(logging.INFO)

torch.autograd.set_detect_anomaly(True)

warnings.filterwarnings("ignore", category=UserWarning)

def make_plots(plot_list, plot_info):
	plt.figure(figsize=(10, 6))
	plt.title(f"{plot_info} Over Time")
	plt.xlabel("Epoch")
	plt.ylabel("AUC/loss")
	plt.plot(range(1, len(plot_list) + 1), plot_list, label=plot_info, marker='o', linestyle='-')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(os.path.join(args.output_dir, f"{plot_info}_{args.softRank}_{args.tn_loss_factor}.png"))
	plt.close()

def get_args_parser():
	parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
	parser.add_argument('--batch_size', default=64, type=int,
						help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
	parser.add_argument('--epochs', default=50, type=int)
	parser.add_argument('--accum_iter', default=1, type=int,
						help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

	# Model parameters
	parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
						help='Name of model to train')
	parser.add_argument('--softRank', type=int, default=76, help='low rank')
	parser.add_argument('--train_all', action='store_true', default=False,
						help='for training/finetuning base model')
	parser.add_argument('--retrain', type=str, default=None,
						help='retraining the model')
	parser.add_argument('--tn_loss_factor', type=float, default=0.005,
						help='scaling factor for tn_loss compared to ce_loss')

	parser.add_argument('--input_size', default=224, type=int,
						help='images input size')

	parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
						help='Drop path rate (default: 0.1)')

	# Optimizer parameters
	parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
						help='Clip gradient norm (default: None, no clipping)')
	parser.add_argument('--weight_decay', type=float, default=0.05,
						help='weight decay (default: 0.05)')

	parser.add_argument('--lr', type=float, default=None, metavar='LR',
						help='learning rate (absolute lr)')
	parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
						help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
	parser.add_argument('--layer_decay', type=float, default=0.75,
						help='layer-wise lr decay from ELECTRA/BEiT')

	parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
						help='lower lr bound for cyclic schedulers that hit 0')

	parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
						help='epochs to warmup LR')

	# Augmentation parameters
	parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
						help='Color jitter factor (enabled only when not using Auto/RandAug)')
	parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
						help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
	parser.add_argument('--smoothing', type=float, default=0.1,
						help='Label smoothing (default: 0.1)')

	# * Random Erase params
	parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
						help='Random erase prob (default: 0.25)')
	parser.add_argument('--remode', type=str, default='pixel',
						help='Random erase mode (default: "pixel")')
	parser.add_argument('--recount', type=int, default=1,
						help='Random erase count (default: 1)')
	parser.add_argument('--resplit', action='store_true', default=False,
						help='Do not random erase first (clean) augmentation split')

	# * Mixup params
	parser.add_argument('--mixup', type=float, default=0,
						help='mixup alpha, mixup enabled if > 0.')
	parser.add_argument('--cutmix', type=float, default=0,
						help='cutmix alpha, cutmix enabled if > 0.')
	parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
						help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
	parser.add_argument('--mixup_prob', type=float, default=1.0,
						help='Probability of performing mixup or cutmix when either/both is enabled')
	parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
						help='Probability of switching to cutmix when both mixup and cutmix enabled')
	parser.add_argument('--mixup_mode', type=str, default='batch',
						help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

	# * Finetuning params
	parser.add_argument('--finetune', default='',
						help='finetune from checkpoint')
	parser.add_argument('--global_pool', action='store_true')
	parser.set_defaults(global_pool=True)
	parser.add_argument('--cls_token', action='store_false', dest='global_pool',
						help='Use class token instead of global pool for classification')

	# Dataset parameters
	parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
						help='dataset path')
	parser.add_argument('--nb_classes', default=1000, type=int,
						help='number of the classification types')

	parser.add_argument('--output_dir', default='./output_dir',
						help='path where to save, empty for no saving')
	parser.add_argument('--log_dir', default='./output_dir',
						help='path where to tensorboard log')
	parser.add_argument('--device', default='cuda',
						help='device to use for training / testing')
	parser.add_argument('--seed', default=0, type=int)
	parser.add_argument('--resume', default='',
						help='resume from checkpoint')

	parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
						help='start epoch')
	parser.add_argument('--eval', action='store_true',
						help='Perform evaluation only')
	parser.add_argument('--eval_path', default='',
						help='evaluation checkpoint')
	parser.add_argument('--dist_eval', action='store_true', default=False,
						help='Enabling distributed evaluation (recommended during training for faster monitor')
	parser.add_argument('--num_workers', default=10, type=int)
	parser.add_argument('--pin_mem', action='store_true',
						help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
	parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
	parser.set_defaults(pin_mem=True)

	# distributed training parameters
	parser.add_argument('--world_size', default=1, type=int,
						help='number of distributed processes')
	parser.add_argument('--local_rank', default=-1, type=int)
	parser.add_argument('--dist_on_itp', action='store_true')
	parser.add_argument('--dist_url', default='env://',
						help='url used to set up distributed training')
	parser.add_argument("--train_list", default=None, type=str, help="file for train list")
	parser.add_argument("--val_list", default=None, type=str, help="file for val list")
	parser.add_argument("--test_list", default=None, type=str, help="file for test list")
	parser.add_argument('--eval_interval', default=10, type=int)
	parser.add_argument('--fixed_lr', action='store_true', default=False)
	parser.add_argument('--vit_dropout_rate', type=float, default=0,
						help='Dropout rate for ViT blocks (default: 0.0)')
	parser.add_argument("--build_timm_transform", action='store_true', default=False)
	parser.add_argument("--aug_strategy", default='default', type=str, help="strategy for data augmentation")
	parser.add_argument("--dataset", default='chestxray', type=str)

	parser.add_argument('--repeated-aug', action='store_true', default=False)

	parser.add_argument("--optimizer", default='adamw', type=str)

	parser.add_argument('--ThreeAugment', action='store_true')  # 3augment

	parser.add_argument('--src', action='store_true')  # simple random crop

	parser.add_argument('--loss_func', default=None, type=str)

	parser.add_argument("--norm_stats", default=None, type=str)

	parser.add_argument("--checkpoint_type", default=None, type=str)

	return parser


def main(args):
	misc.init_distributed_mode(args)

	print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
	print("{}".format(args).replace(', ', ',\n'))

	device = torch.device(args.device)

	# fix the seed for reproducibility
	seed = args.seed + misc.get_rank()
	torch.manual_seed(seed)
	np.random.seed(seed)

	cudnn.benchmark = True

	dataset_train = build_dataset_chest_xray(split='train', args=args)
	# dataset_val = build_dataset_chest_xray(split='val', args=args)
	dataset_test = build_dataset_chest_xray(split='test', args=args)

	if True:  # args.distributed:
		num_tasks = misc.get_world_size()
		global_rank = misc.get_rank()
		if args.repeated_aug:
			sampler_train = RASampler(
				dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
			)
		else:
			sampler_train = torch.utils.data.DistributedSampler(
				dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
			)
		print("Sampler_train = %s" % str(sampler_train))
		if args.dist_eval:
			if len(dataset_test) % num_tasks != 0:
				print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
					  'This will slightly alter validation results as extra duplicate entries are added to achieve '
					  'equal num of samples per-process.')
			# sampler_val = torch.utils.data.DistributedSampler(
			#     dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
			sampler_test = torch.utils.data.DistributedSampler(
				dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)
		else:
			# sampler_val = torch.utils.data.SequentialSampler(dataset_val)
			sampler_test = torch.utils.data.SequentialSampler(dataset_test)
	else:
		sampler_train = torch.utils.data.RandomSampler(dataset_train)
		# sampler_val = torch.utils.data.SequentialSampler(dataset_val)
		sampler_test = torch.utils.data.SequentialSampler(dataset_test)

	if global_rank == 0 and args.log_dir is not None and not args.eval:
		os.makedirs(args.log_dir, exist_ok=True)
		log_writer = SummaryWriter(log_dir=args.log_dir)
	else:
		log_writer = None

	data_loader_train = torch.utils.data.DataLoader(
		dataset_train, sampler=sampler_train,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		pin_memory=args.pin_mem,
		drop_last=False,
	)

	data_loader_test = torch.utils.data.DataLoader(
		dataset_test, sampler=sampler_test,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		pin_memory=args.pin_mem,
		drop_last=False
	)

	mixup_fn = None
	mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
	n = len(dataset_train)
	if mixup_active:
		print("Mixup is activated!")
		mixup_fn = Mixup(
			mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
			prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
			label_smoothing=args.smoothing, num_classes=args.nb_classes)
	if 'vit' in args.model:
		model = models_vit.__dict__[args.model](
			img_size=args.input_size,
			num_classes=args.nb_classes,
			drop_rate=args.vit_dropout_rate,
			drop_path_rate=args.drop_path,
			global_pool=args.global_pool,
		)
		if 'vit_small' in args.model:
			dim = 384
		else:
			dim=768

	elif 'densenet' in args.model:
		model = models_densenet.__dict__[args.model](num_classes=args.nb_classes)
		dim = 1024
	elif 'resnet' in args.model:
		model = models_resnet.__dict__[args.model](num_classes=args.nb_classes)
		dim = 2048
	else:
		raise NotImplementedError


	if args.finetune and not args.eval:
		if 'vit' in args.model:
			checkpoint = torch.load(args.finetune, map_location='cpu')

			print("Load pre-trained checkpoint from: %s" % args.finetune)
			checkpoint_model = checkpoint['model']
			state_dict = model.state_dict()
			for k in ['head.weight', 'head.bias']:
				if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
					print(f"Removing key {k} from pretrained checkpoint")
					del checkpoint_model[k]
			if args.global_pool:
				for k in ['fc_norm.weight', 'fc_norm.bias']:
					try:
						del checkpoint_model[k]
					except:
						pass


			# interpolate position embedding
			interpolate_pos_embed(model, checkpoint_model)

			# load pre-trained model
			msg = model.load_state_dict(checkpoint_model, strict=False)
			print(msg)


			# if args.global_pool:
			#     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
			# else:
			#     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

			# manually initialize fc layer
			trunc_normal_(model.head.weight, std=2e-5)
		elif 'densenet' in args.model or 'resnet' in args.model:
			checkpoint = torch.load(args.finetune, map_location='cpu')
			print("Load pre-trained checkpoint from: %s" % args.finetune)
			if 'state_dict' in checkpoint.keys():
				checkpoint_model = checkpoint['state_dict']
			elif 'model' in checkpoint.keys():
				checkpoint_model = checkpoint['model']
			else:
				checkpoint_model = checkpoint
			if args.checkpoint_type == 'smp_encoder':
				state_dict = checkpoint_model

				new_state_dict = OrderedDict()

				for key, value in state_dict.items():
					if 'model.encoder.' in key:
						new_key = key.replace('model.encoder.', '')
						new_state_dict[new_key] = value
				checkpoint_model = new_state_dict
			msg = model.load_state_dict(checkpoint_model, strict=False)
			print(msg)

	if args.retrain:
		checkpoint = torch.load(args.retrain)
		if 'state_dict' in checkpoint.keys():
			checkpoint_model = checkpoint['state_dict']
		elif 'model' in checkpoint.keys():
			checkpoint_model = checkpoint['model']
		else:
			checkpoint_model = checkpoint
		
		if 'densenet' in args.model:
			checkpoint_model_temp = checkpoint_model.copy()

			for k, v in checkpoint_model_temp.items():
				if 'classifier.0.' in k:
					new_k = k.replace('classifier.0.', 'classifier.')
					checkpoint_model[new_k] = v
					del checkpoint_model[k]

		if 'resnet' in args.model:
			checkpoint_model_temp = checkpoint_model.copy()

			for k, v in checkpoint_model_temp.items():
				if 'fc.0.' in k:
					new_k = k.replace('fc.0.', 'fc.')
					checkpoint_model[new_k] = v
					del checkpoint_model[k]
	
		model.load_state_dict(checkpoint_model, strict=False)
		
	model.to(device)

	if args.eval:
		checkpoint = torch.load(args.eval_path)
		if 'state_dict' in checkpoint.keys():
			checkpoint_model = checkpoint['state_dict']
		elif 'model' in checkpoint.keys():
			checkpoint_model = checkpoint['model']
		else:
			checkpoint_model = checkpoint
		
		if 'densenet' in args.model:
			checkpoint_model_temp = checkpoint_model.copy()

			for k, v in checkpoint_model_temp.items():
				if 'classifier.0.' in k:
					new_k = k.replace('classifier.0.', 'classifier.')
					checkpoint_model[new_k] = v
					del checkpoint_model[k]
		
		if 'resnet' in args.model:
			checkpoint_model_temp = checkpoint_model.copy()

			for k, v in checkpoint_model_temp.items():
				if 'fc.0.' in k:
					new_k = k.replace('fc.0.', 'fc.')
					checkpoint_model[new_k] = v
					del checkpoint_model[k]

		model.load_state_dict(checkpoint_model, strict=True)
		test_stats, loss = evaluate_chestxray(data_loader_test, model, device, args)
		print(f"Average AUC of the network on the test set images: {test_stats['auc_avg']:.4f}")
		if args.dataset == 'covidx':
			print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
		exit(0)

	model_without_ddp = model
	n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

	print("Model = %s" % str(model_without_ddp))
	print('number of params (M): %.2f' % (n_parameters / 1.e6))

	eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
	
	if args.lr is None:  # only base_lr is specified
		args.lr = args.blr * eff_batch_size / 256

	print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
	print("actual lr: %.2e" % args.lr)

	print("accumulate grad iterations: %d" % args.accum_iter)
	print("effective batch size: %d" % eff_batch_size)

	if args.distributed:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
		model_without_ddp = model.module

	# build optimizer with layer-wise lr decay (lrd)
	if 'vit' in args.model:
		param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
			no_weight_decay_list=model_without_ddp.no_weight_decay(),
			layer_decay=args.layer_decay
		)
	else:
		param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)

	if args.optimizer == 'adamw':
		optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
	elif args.optimizer == 'fusedlamb':
		optimizer = FusedAdam(param_groups, lr=args.lr)
	loss_scaler = NativeScaler()

	if args.dataset == 'chestxray':
		if mixup_fn is not None:
			# smoothing is handled with mixup label transform
			criterion = SoftTargetBinaryCrossEntropy()
		else:
			criterion = torch.nn.BCEWithLogitsLoss()
			# if args.smoothing > 0.:
			# 	criterion = BinaryCrossEntropy(smoothing=args.smoothing)
	elif args.dataset == 'covidx':
		criterion = torch.nn.CrossEntropyLoss()
	elif args.dataset == 'node21':
		if args.loss_func == 'bce':
			criterion = torch.nn.BCEWithLogitsLoss()
		elif args.loss_func is None:
			criterion = torch.nn.CrossEntropyLoss()
	elif args.dataset == 'chexpert':
		criterion = losses.CrossEntropyLoss()
	else:
		raise NotImplementedError
	# elif args.smoothing > 0.:
	#     criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

	# if
	# criterion = torch.nn.BCEWithLogitsLoss()

	print("criterion = %s" % str(criterion))

	misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

	print(f"Start training for {args.epochs} epochs")
	start_time = time.time()
	max_accuracy = 0.0
	max_auc = 0.0
	best_auc = -1000000
	best_loss = 1000000000
	auc_plot = []
	test_loss_plot = []
	train_loss_plot = []
	ce_loss_plot = []
	tn_loss_plot = []

	if args.train_list is not None:
		perc = args.train_list.split('/')[-1].split('_')[-1].split('.')[0]
	else:
		perc = ''
	split = perc if perc != '' else '100'

	for epoch in range(args.start_epoch, args.epochs):
		U, V = torch.empty((0)).cuda(dist.get_rank()), torch.empty((0)).cuda(dist.get_rank())

		features, indexes = evaluate_chestxray_before(
			data_loader_train, model, device, args)
		
		dist.barrier()

		feature_list = [torch.zeros_like(features)
							for _ in range(dist.get_world_size())]
		index_list = [torch.zeros_like(indexes)
						for _ in range(dist.get_world_size())]

		if args.rank == 0:
			dist.gather(features, gather_list=feature_list,
						group=dist.group.WORLD)
			dist.gather(indexes, gather_list=index_list,
						group=dist.group.WORLD)
		else:
			dist.gather(features, group=dist.group.WORLD)
			dist.gather(indexes, group=dist.group.WORLD)
		
		dist.barrier()
		
		U, V = torch.empty((n, dim)).cuda(), torch.empty((dim, dim)).cuda()
		if args.rank == 0:
			new_features = torch.empty((0)).cuda()
			new_indexes = torch.empty((0)).cuda()
			for idx1 in index_list:
				new_indexes = torch.cat((new_indexes, idx1), dim=0)

			for feature1 in feature_list:
				new_features = torch.cat((new_features, feature1), dim=0)
		
			sorted_indexes = torch.argsort(new_indexes)
			new_features = new_features[sorted_indexes]

			new_features = new_features.cpu().numpy()

			U, S, V = scipy.linalg.svd(new_features, full_matrices=False)
			U, V = torch.tensor(U).cuda().contiguous(), torch.tensor(V).cuda().contiguous()
		dist.barrier()

		dist.broadcast(U, src=0, async_op=False, group=dist.group.WORLD)
		dist.broadcast(V, src=0, async_op=False, group=dist.group.WORLD)	

		if args.distributed:
			data_loader_train.sampler.set_epoch(epoch)
		train_stats = train_one_epoch(
			model, criterion, data_loader_train,
			optimizer, device, epoch, loss_scaler,
			args.clip_grad, mixup_fn,
			log_writer=log_writer, U=U, V=V,
			args=args
		)

		if args.output_dir and (epoch % args.eval_interval == 0 or epoch + 1 == args.epochs):
			misc.save_model(
				args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
				loss_scaler=loss_scaler, epoch=epoch, name=f'latest_{args.softRank}_{args.tn_loss_factor}_{args.epochs}_{split}_%')

			test_stats, loss = evaluate_chestxray(data_loader_test, model, device, args)
			print(f"Average AUC on the test set images: {test_stats['auc_avg']:.4f}")
			max_auc = max(max_auc, test_stats['auc_avg'])
			auc_plot.append(test_stats['auc_avg'])
			test_loss_plot.append(loss.item())
			train_loss_plot.append(train_stats['loss'])
			ce_loss_plot.append(train_stats['ce_loss'])
			tn_loss_plot.append(train_stats['tn_loss'])
			print(f'Max Average AUC: {max_auc:.4f}', {max_auc})

			if loss < best_loss:
				best_loss = loss
				misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
								loss_scaler=loss_scaler, epoch=epoch, name=f'best_loss_{args.softRank}_{args.tn_loss_factor}_{args.epochs}_{split}%')

			if best_auc <= test_stats['auc_avg']:
				best_auc = test_stats['auc_avg']
				misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
								loss_scaler=loss_scaler, epoch=epoch, name=f'best_auc_{args.softRank}_{args.tn_loss_factor}_{args.epochs}_{split}%')

			if args.dataset == 'covidx':
				print(f"Accuracy of the network on the {len(data_loader_test)} test images: {test_stats['acc1']:.1f}%")
				max_accuracy = max(max_accuracy, test_stats["acc1"])
				print(f'Max accuracy: {max_accuracy:.2f}%')

			if log_writer is not None:
				log_writer.add_scalar('perf/auc_avg', test_stats['auc_avg'], epoch)
				log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

			log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
							**{f'test_{k}': v for k, v in test_stats.items()},
							'epoch': epoch,
							'n_parameters': n_parameters}

			if args.output_dir and misc.is_main_process():
				if log_writer is not None:
					log_writer.flush()
				with open(os.path.join(args.output_dir, f"log_{args.softRank}_{args.tn_loss_factor}_{args.epochs}_{split}%.txt"), mode="a", encoding="utf-8") as f:
					f.write(json.dumps(log_stats) + "\n")

	make_plots(auc_plot, 'auc')	
	make_plots(test_loss_plot, 'test_loss')	
	make_plots(train_loss_plot, 'train_loss')	
	make_plots(ce_loss_plot, 'ce_loss')	
	make_plots(tn_loss_plot, 'tn_loss')	
	
	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
	args = get_args_parser()
	args = args.parse_args()
	setup_default_logging()
	if args.output_dir:
		Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	main(args)
