import argparse
import logging
import time
import math

import numpy as np
import torch.utils.data

import torch.nn.functional as F

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation, grasp
from utils.visualisation.plot import save_results

from fxpmath import Fxp

from inference_fxp import param_convert, loss

logging.basicConfig(level=logging.INFO)

def get_max_value(test_data):
	"""
	This function returns the greatest value inside the test_data
	"""
	greatest_value = -1

	for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
		local_value = torch.max(x)
		if local_value > greatest_value:
			greatest_value = local_value
	return greatest_value


def parse_args():
	parser = argparse.ArgumentParser(description='Evaluate networks')

	# Network
	parser.add_argument('--network', metavar='N', type=str, nargs='+',
						help='Path to saved networks to evaluate')

	# Dataset
	parser.add_argument('--dataset', type=str,
						help='Dataset Name ("cornell" or "jaquard")')
	parser.add_argument('--dataset-path', type=str,
						help='Path to dataset')
	parser.add_argument('--use-depth', type=int, default=1,
						help='Use Depth image for evaluation (1/0)')
	parser.add_argument('--use-rgb', type=int, default=1,
						help='Use RGB image for evaluation (1/0)')
	parser.add_argument('--augment', action='store_true',
						help='Whether data augmentation should be applied')
	parser.add_argument('--split', type=float, default=0.9,
						help='Fraction of data for training (remainder is validation)')
	parser.add_argument('--ds-shuffle', action='store_true', default=False,
						help='Shuffle the dataset')
	parser.add_argument('--ds-rotate', type=float, default=0.0,
						help='Shift the start point of the dataset to use a different test/train split')
	parser.add_argument('--num-workers', type=int, default=8,
						help='Dataset workers')

	# Evaluation
	parser.add_argument('--n-grasps', type=int, default=1,
						help='Number of grasps to consider per image')
	parser.add_argument('--iou-threshold', type=float, default=0.25,
						help='Threshold for IOU matching')
	parser.add_argument('--iou-eval', action='store_true',
						help='Compute success based on IoU metric.')
	parser.add_argument('--jacquard-output', action='store_true',
						help='Jacquard-dataset style output')

	# Misc.
	parser.add_argument('--vis', action='store_true',
						help='Visualise the network output')
	parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
						help='Force code to run in CPU mode')
	parser.add_argument('--random-seed', type=int, default=123,
						help='Random seed for numpy')

	args = parser.parse_args()

	if args.jacquard_output and args.dataset != 'jacquard':
		raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
	if args.jacquard_output and args.augment:
		raise ValueError('--jacquard-output can not be used with data augmentation.')

	return args


if __name__ == '__main__':
	args = parse_args()

	# Get the compute device
	# device = get_device(args.force_cpu)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('Using device:', device)
	print()

	#Additional Info when using cuda
	if device.type == 'cuda':
		print(torch.cuda.get_device_name(0))
		print('Memory Usage:')
		print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
		print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')



	# Load Dataset
	logging.info('Loading {} Dataset...'.format(args.dataset.title()))
	Dataset = get_dataset(args.dataset)
	test_dataset = Dataset(args.dataset_path,
						   ds_rotate=args.ds_rotate,
						   random_rotate=args.augment,
						   random_zoom=args.augment,
						   include_depth=args.use_depth,
						   include_rgb=args.use_rgb)

	indices = list(range(test_dataset.length))
	split = int(np.floor(args.split * test_dataset.length))
	if args.ds_shuffle:
		np.random.seed(args.random_seed)
		np.random.shuffle(indices)
	val_indices = indices[split:]
	val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
	logging.info('Validation size: {}'.format(len(val_indices)))

	test_data = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=1,
		num_workers=args.num_workers,
		sampler=val_sampler
	)
	test_features, test_labels, a, b, c= next(iter(test_data))
	logging.info('Done')
	print("type of test_dataset = ",type(test_data))
	print("type of test_data = ",test_labels[0].size())
	greatest_value = get_max_value(test_data)
	print ("greatest_value = ",greatest_value)
	for network in args.network:
		logging.info('\nEvaluating model {}'.format(network))

		# Load Network
		net = torch.load(network)

		model_dict = net.state_dict()
