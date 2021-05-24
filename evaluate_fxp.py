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
    device = get_device(args.force_cpu)

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

    for network in args.network:
        logging.info('\nEvaluating model {}'.format(network))

        # Load Network
        net = torch.load(network)
        x_0,x_1,x_2,x_3 = net(test_features)
        print(x_0.size())
        print(x_1.size())

        param_lenth = 8

        model_dict = net.state_dict()
        conv1_weight = model_dict['conv1.weight'].numpy()
        conv1_bias = model_dict['conv1.bias'].numpy()
        bn1_weight = model_dict['bn1.weight'].numpy()
        bn1_bias = model_dict['bn1.bias'].numpy()
        bn1_running_mean = model_dict['bn1.running_mean'].numpy()
        bn1_running_var = model_dict['bn1.running_var'].numpy()

        conv2_weight = model_dict['conv2.weight'].numpy()
        conv2_bias = model_dict['conv2.bias'].numpy()
        bn2_weight = model_dict['bn2.weight'].numpy()
        bn2_bias = model_dict['bn2.bias'].numpy()
        bn2_running_mean = model_dict['bn2.running_mean'].numpy()
        bn2_running_var = model_dict['bn2.running_var'].numpy()

        conv3_weight = model_dict['conv3.weight'].numpy()
        conv3_bias = model_dict['conv3.bias'].numpy()
        bn3_weight = model_dict['bn3.weight'].numpy()
        bn3_bias = model_dict['bn3.bias'].numpy()
        bn3_running_mean = model_dict['bn3.running_mean'].numpy()
        bn3_running_var = model_dict['bn3.running_var'].numpy()

        res1_conv1_weight = model_dict['res1.conv1.weight'].numpy()
        res1_conv1_bias = model_dict['res1.conv1.bias'].numpy()
        res1_bn1_weight = model_dict['res1.bn1.weight'].numpy()
        res1_bn1_bias = model_dict['res1.bn1.bias'].numpy()
        res1_bn1_running_mean = model_dict['res1.bn1.running_mean'].numpy()
        res1_bn1_running_var = model_dict['res1.bn1.running_var'].numpy()
        res1_conv2_weight = model_dict['res1.conv2.weight'].numpy()
        res1_conv2_bias = model_dict['res1.conv2.bias'].numpy()
        res1_bn2_weight = model_dict['res1.bn2.weight'].numpy()
        res1_bn2_bias = model_dict['res1.bn2.bias'].numpy()
        res1_bn2_running_mean = model_dict['res1.bn2.running_mean'].numpy()
        res1_bn2_running_var = model_dict['res1.bn2.running_var'].numpy()

        res2_conv1_weight = model_dict['res2.conv1.weight'].numpy()
        res2_conv1_bias = model_dict['res2.conv1.bias'].numpy()
        res2_bn1_weight = model_dict['res2.bn1.weight'].numpy()
        res2_bn1_bias = model_dict['res2.bn1.bias'].numpy()
        res2_bn1_running_mean = model_dict['res2.bn1.running_mean'].numpy()
        res2_bn1_running_var = model_dict['res2.bn1.running_var'].numpy()
        res2_conv2_weight = model_dict['res2.conv2.weight'].numpy()
        res2_conv2_bias = model_dict['res2.conv2.bias'].numpy()
        res2_bn2_weight = model_dict['res2.bn2.weight'].numpy()
        res2_bn2_bias = model_dict['res2.bn2.bias'].numpy()
        res2_bn2_running_mean = model_dict['res2.bn2.running_mean'].numpy()
        res2_bn2_running_var = model_dict['res2.bn2.running_var'].numpy()

        res3_conv1_weight = model_dict['res3.conv1.weight'].numpy()
        res3_conv1_bias = model_dict['res3.conv1.bias'].numpy()
        res3_bn1_weight = model_dict['res3.bn1.weight'].numpy()
        res3_bn1_bias = model_dict['res3.bn1.bias'].numpy()
        res3_bn1_running_mean = model_dict['res3.bn1.running_mean'].numpy()
        res3_bn1_running_var = model_dict['res3.bn1.running_var'].numpy()
        res3_conv2_weight = model_dict['res3.conv2.weight'].numpy()
        res3_conv2_bias = model_dict['res3.conv2.bias'].numpy()
        res3_bn2_weight = model_dict['res3.bn2.weight'].numpy()
        res3_bn2_bias = model_dict['res3.bn2.bias'].numpy()
        res3_bn2_running_mean = model_dict['res3.bn2.running_mean'].numpy()
        res3_bn2_running_var = model_dict['res3.bn2.running_var'].numpy()

        res4_conv1_weight = model_dict['res4.conv1.weight'].numpy()
        res4_conv1_bias = model_dict['res4.conv1.bias'].numpy()
        res4_bn1_weight = model_dict['res4.bn1.weight'].numpy()
        res4_bn1_bias = model_dict['res4.bn1.bias'].numpy()
        res4_bn1_running_mean = model_dict['res4.bn1.running_mean'].numpy()
        res4_bn1_running_var = model_dict['res4.bn1.running_var'].numpy()
        res4_conv2_weight = model_dict['res4.conv2.weight'].numpy()
        res4_conv2_bias = model_dict['res4.conv2.bias'].numpy()
        res4_bn2_weight = model_dict['res4.bn2.weight'].numpy()
        res4_bn2_bias = model_dict['res4.bn2.bias'].numpy()
        res4_bn2_running_mean = model_dict['res4.bn2.running_mean'].numpy()
        res4_bn2_running_var = model_dict['res4.bn2.running_var'].numpy()

        res5_conv1_weight = model_dict['res5.conv1.weight'].numpy()
        res5_conv1_bias = model_dict['res5.conv1.bias'].numpy()
        res5_bn1_weight = model_dict['res5.bn1.weight'].numpy()
        res5_bn1_bias = model_dict['res5.bn1.bias'].numpy()
        res5_bn1_running_mean = model_dict['res5.bn1.running_mean'].numpy()
        res5_bn1_running_var = model_dict['res5.bn1.running_var'].numpy()
        res5_conv2_weight = model_dict['res5.conv2.weight'].numpy()
        res5_conv2_bias = model_dict['res5.conv2.bias'].numpy()
        res5_bn2_weight = model_dict['res5.bn2.weight'].numpy()
        res5_bn2_bias = model_dict['res5.bn2.bias'].numpy()
        res5_bn2_running_mean = model_dict['res5.bn2.running_mean'].numpy()
        res5_bn2_running_var = model_dict['res5.bn2.running_var'].numpy()

        conv4_weight = model_dict['conv4.weight'].numpy()
        conv4_bias = model_dict['conv4.bias'].numpy()
        bn4_weight = model_dict['bn4.weight'].numpy()
        bn4_bias = model_dict['bn4.bias'].numpy()
        bn4_running_mean = model_dict['bn4.running_mean'].numpy()
        bn4_running_var = model_dict['bn4.running_var'].numpy()

        conv5_weight = model_dict['conv5.weight'].numpy()
        conv5_bias = model_dict['conv5.bias'].numpy()
        bn5_weight = model_dict['bn5.weight'].numpy()
        bn5_bias = model_dict['bn5.bias'].numpy()
        bn5_running_mean = model_dict['bn5.running_mean'].numpy()
        bn5_running_var = model_dict['bn5.running_var'].numpy()

        conv6_weight = model_dict['conv6.weight'].numpy()
        conv6_bias = model_dict['conv6.bias'].numpy()

        pos_weight = model_dict['pos_output.weight'].numpy()
        pos_bias = model_dict['pos_output.bias'].numpy()
        cos_weight = model_dict['cos_output.weight'].numpy()
        cos_bias = model_dict['cos_output.bias'].numpy()
        sin_weight = model_dict['sin_output.weight'].numpy()
        sin_bias = model_dict['sin_output.bias'].numpy()
        width_weight = model_dict['width_output.weight'].numpy()
        width_bias = model_dict['width_output.bias'].numpy()

        for i in range(len(bn1_weight)):
            bn1_weight[i] = bn1_weight[i]/math.sqrt(bn1_running_var[i] + 0.00001)
            bn1_bias[i] = bn1_bias[i] - bn1_running_mean[i]/math.sqrt(bn1_running_var[i] + 0.00001)
            bn1_running_var[i] = 0.99999
            bn1_running_mean[i] = 0

        for i in range(len(bn2_weight)):
            bn2_weight[i] = bn2_weight[i]/math.sqrt(bn2_running_var[i]+0.00001)
            bn2_bias[i] = bn2_bias[i] - bn2_running_mean[i]/math.sqrt(bn2_running_var[i]+0.00001)
            bn2_running_var[i] = 0.99999
            bn2_running_mean[i] = 0

        for i in range(len(bn3_weight)):
            bn3_weight[i] = bn3_weight[i]/math.sqrt(bn3_running_var[i]+0.00001)
            bn3_bias[i] = bn3_bias[i] - bn3_running_mean[i]/math.sqrt(bn3_running_var[i]+0.00001)
            bn3_running_var[i] = 0.99999
            bn3_running_mean[i] = 0

        for i in range(len(res1_bn1_weight)):
            res1_bn1_weight[i] = res1_bn1_weight[i]/math.sqrt(res1_bn1_running_var[i]+0.00001)
            res1_bn1_bias[i] = res1_bn1_bias[i] - res1_bn1_running_mean[i]/math.sqrt(res1_bn1_running_var[i]+0.00001)
            res1_bn1_running_var[i] = 0.99999
            res1_bn1_running_mean[i] = 0
            res1_bn2_weight[i] = res1_bn2_weight[i]/math.sqrt(res1_bn2_running_var[i]+0.00001)
            res1_bn2_bias[i] = res1_bn2_bias[i] - res1_bn2_running_mean[i]/math.sqrt(res1_bn2_running_var[i]+0.00001)
            res1_bn2_running_var[i] = 0.99999
            res1_bn2_running_mean[i] = 0
            
            res2_bn1_weight[i] = res2_bn1_weight[i]/math.sqrt(res2_bn1_running_var[i]+0.00001)
            res2_bn1_bias[i] = res2_bn1_bias[i] - res2_bn1_running_mean[i]/math.sqrt(res2_bn1_running_var[i]+0.00001)
            res2_bn1_running_var[i] = 0.99999
            res2_bn1_running_mean[i] = 0
            res2_bn2_weight[i] = res2_bn2_weight[i]/math.sqrt(res2_bn2_running_var[i]+0.00001)
            res2_bn2_bias[i] = res2_bn2_bias[i] - res2_bn2_running_mean[i]/math.sqrt(res2_bn2_running_var[i]+0.00001)
            res2_bn2_running_var[i] = 0.99999
            res2_bn2_running_mean[i] = 0

            res3_bn1_weight[i] = res3_bn1_weight[i]/math.sqrt(res3_bn1_running_var[i]+0.00001)
            res3_bn1_bias[i] = res3_bn1_bias[i] - res3_bn1_running_mean[i]/math.sqrt(res3_bn1_running_var[i]+0.00001)
            res3_bn1_running_var[i] = 0.99999
            res3_bn1_running_mean[i] = 0
            res3_bn2_weight[i] = res3_bn2_weight[i]/math.sqrt(res3_bn2_running_var[i]+0.00001)
            res3_bn2_bias[i] = res3_bn2_bias[i] - res3_bn2_running_mean[i]/math.sqrt(res3_bn2_running_var[i]+0.00001)
            res3_bn2_running_var[i] = 0.99999
            res3_bn2_running_mean[i] = 0

            res4_bn1_weight[i] = res4_bn1_weight[i]/math.sqrt(res4_bn1_running_var[i]+0.00001)
            res4_bn1_bias[i] = res4_bn1_bias[i] - res4_bn1_running_mean[i]/math.sqrt(res4_bn1_running_var[i]+0.00001)
            res4_bn1_running_var[i] = 0.99999
            res4_bn1_running_mean[i] = 0
            res4_bn2_weight[i] = res4_bn2_weight[i]/math.sqrt(res4_bn2_running_var[i]+0.00001)
            res4_bn2_bias[i] = res4_bn2_bias[i] - res4_bn2_running_mean[i]/math.sqrt(res4_bn2_running_var[i]+0.00001)
            res4_bn2_running_var[i] = 0.99999
            res4_bn2_running_mean[i] = 0

            res5_bn1_weight[i] = res5_bn1_weight[i]/math.sqrt(res5_bn1_running_var[i]+0.00001)
            res5_bn1_bias[i] = res5_bn1_bias[i] - res5_bn1_running_mean[i]/math.sqrt(res5_bn1_running_var[i]+0.00001)
            res5_bn1_running_var[i] = 0.99999
            res5_bn1_running_mean[i] = 0
            res5_bn2_weight[i] = res5_bn2_weight[i]/math.sqrt(res5_bn2_running_var[i]+0.00001)
            res5_bn2_bias[i] = res5_bn2_bias[i] - res5_bn2_running_mean[i]/math.sqrt(res5_bn2_running_var[i]+0.00001)
            res5_bn2_running_var[i] = 0.99999
            res5_bn2_running_mean[i] = 0

        for i in range(len(bn4_weight)):
            bn4_weight[i] = bn4_weight[i]/math.sqrt(bn4_running_var[i]+0.00001)
            bn4_bias[i] = bn4_bias[i] - bn4_running_mean[i]/math.sqrt(bn4_running_var[i]+0.00001)
            bn4_running_var[i] = 0.99999
            bn4_running_mean[i] = 0

        for i in range(len(bn5_weight)):
            bn5_weight[i] = bn5_weight[i]/math.sqrt(bn5_running_var[i]+0.00001)
            bn5_bias[i] = bn5_bias[i] - bn5_running_mean[i]/math.sqrt(bn5_running_var[i]+0.00001)
            bn5_running_var[i] = 0.99999
            bn5_running_mean[i] = 0

        conv1_weight = param_convert(conv1_weight,param_lenth)
        conv1_bias = param_convert(conv1_bias,param_lenth)
        bn1_weight = param_convert(bn1_weight,param_lenth)
        bn1_bias = param_convert(bn1_bias,param_lenth)
        bn1_running_mean = param_convert(bn1_running_mean,param_lenth)
        bn1_running_var = param_convert(bn1_running_var,param_lenth)

        conv2_weight = param_convert(conv2_weight,param_lenth)
        conv2_bias = param_convert(conv2_bias,param_lenth)
        bn2_weight = param_convert(bn2_weight,param_lenth)
        bn2_bias = param_convert(bn2_bias,param_lenth)
        bn2_running_mean = param_convert(bn2_running_mean,param_lenth)
        bn2_running_var = param_convert(bn2_running_var,param_lenth)

        conv3_weight = param_convert(conv3_weight,param_lenth)
        conv3_bias = param_convert(conv3_bias,param_lenth)
        bn3_weight = param_convert(bn3_weight,param_lenth)
        bn3_bias = param_convert(bn3_bias,param_lenth)
        bn3_running_mean = param_convert(bn3_running_mean,param_lenth)
        bn3_running_var = param_convert(bn3_running_var,param_lenth)

        res1_conv1_weight = param_convert(res1_conv1_weight,param_lenth)
        res1_conv1_bias = param_convert(res1_conv1_bias,param_lenth)
        res1_bn1_weight = param_convert(res1_bn1_weight,param_lenth)
        res1_bn1_bias = param_convert(res1_bn1_bias,param_lenth)
        res1_bn1_running_mean = param_convert(res1_bn1_running_mean,param_lenth)
        res1_bn1_running_var = param_convert(res1_bn1_running_var,param_lenth)
        res1_conv2_weight = param_convert(res1_conv2_weight,param_lenth)
        res1_conv2_bias = param_convert(res1_conv2_bias,param_lenth)
        res1_bn2_weight = param_convert(res1_bn2_weight,param_lenth)
        res1_bn2_bias = param_convert(res1_bn2_bias,param_lenth)
        res1_bn2_running_mean = param_convert(res1_bn2_running_mean,param_lenth)
        res1_bn2_running_var = param_convert(res1_bn2_running_var,param_lenth)

        res2_conv1_weight =     param_convert(res2_conv1_weight,param_lenth)
        res2_conv1_bias =       param_convert(res2_conv1_bias,param_lenth)
        res2_bn1_weight =       param_convert(res2_bn1_weight,param_lenth)
        res2_bn1_bias =         param_convert(res2_bn1_bias,param_lenth)
        res2_bn1_running_mean = param_convert(res2_bn1_running_mean,param_lenth)
        res2_bn1_running_var =  param_convert(res2_bn1_running_var,param_lenth)
        res2_conv2_weight =     param_convert(res2_conv2_weight,param_lenth)
        res2_conv2_bias =       param_convert(res2_conv2_bias,param_lenth)
        res2_bn2_weight =       param_convert(res2_bn2_weight,param_lenth)
        res2_bn2_bias =         param_convert(res2_bn2_bias,param_lenth)
        res2_bn2_running_mean = param_convert(res2_bn2_running_mean,param_lenth)
        res2_bn2_running_var =  param_convert(res2_bn2_running_var,param_lenth)

        res3_conv1_weight =     param_convert(res3_conv1_weight,param_lenth)
        res3_conv1_bias =       param_convert(res3_conv1_bias,param_lenth)
        res3_bn1_weight =       param_convert(res3_bn1_weight,param_lenth)
        res3_bn1_bias =         param_convert(res3_bn1_bias,param_lenth)
        res3_bn1_running_mean = param_convert(res3_bn1_running_mean,param_lenth)
        res3_bn1_running_var =  param_convert(res3_bn1_running_var,param_lenth)
        res3_conv2_weight =     param_convert(res3_conv2_weight,param_lenth)
        res3_conv2_bias =       param_convert(res3_conv2_bias,param_lenth)
        res3_bn2_weight =       param_convert(res3_bn2_weight,param_lenth)
        res3_bn2_bias =         param_convert(res3_bn2_bias,param_lenth)
        res3_bn2_running_mean = param_convert(res3_bn2_running_mean,param_lenth)
        res3_bn2_running_var =  param_convert(res3_bn2_running_var,param_lenth)

        res4_conv1_weight =     param_convert(res4_conv1_weight,param_lenth)
        res4_conv1_bias =       param_convert(res4_conv1_bias,param_lenth)
        res4_bn1_weight =       param_convert(res4_bn1_weight,param_lenth)
        res4_bn1_bias =         param_convert(res4_bn1_bias,param_lenth)
        res4_bn1_running_mean = param_convert(res4_bn1_running_mean,param_lenth)
        res4_bn1_running_var =  param_convert(res4_bn1_running_var,param_lenth)
        res4_conv2_weight =     param_convert(res4_conv2_weight,param_lenth)
        res4_conv2_bias =       param_convert(res4_conv2_bias,param_lenth)
        res4_bn2_weight =       param_convert(res4_bn2_weight,param_lenth)
        res4_bn2_bias =         param_convert(res4_bn2_bias,param_lenth)
        res4_bn2_running_mean = param_convert(res4_bn2_running_mean,param_lenth)
        res4_bn2_running_var =  param_convert(res4_bn2_running_var,param_lenth)

        res5_conv1_weight =     param_convert(res5_conv1_weight,param_lenth)
        res5_conv1_bias =       param_convert(res5_conv1_bias,param_lenth)
        res5_bn1_weight =       param_convert(res5_bn1_weight,param_lenth)
        res5_bn1_bias =         param_convert(res5_bn1_bias,param_lenth)
        res5_bn1_running_mean = param_convert(res5_bn1_running_mean,param_lenth)
        res5_bn1_running_var =  param_convert(res5_bn1_running_var,param_lenth)
        res5_conv2_weight =     param_convert(res5_conv2_weight,param_lenth)
        res5_conv2_bias =       param_convert(res5_conv2_bias,param_lenth)
        res5_bn2_weight =       param_convert(res5_bn2_weight,param_lenth)
        res5_bn2_bias =         param_convert(res5_bn2_bias,param_lenth)
        res5_bn2_running_mean = param_convert(res5_bn2_running_mean,param_lenth)
        res5_bn2_running_var =  param_convert(res5_bn2_running_var,param_lenth)

        conv4_weight = param_convert(conv4_weight,param_lenth)
        conv4_bias = param_convert(conv4_bias,param_lenth)
        bn4_weight = param_convert(bn4_weight,param_lenth)
        bn4_bias = param_convert(bn4_bias,param_lenth)
        bn4_running_mean = param_convert(bn4_running_mean,param_lenth)
        bn4_running_var = param_convert(bn4_running_var,param_lenth)

        conv5_weight = param_convert(conv5_weight,param_lenth)
        conv5_bias = param_convert(conv5_bias,param_lenth)
        bn5_weight = param_convert(bn5_weight,param_lenth)
        bn5_bias = param_convert(bn5_bias,param_lenth)
        bn5_running_mean = param_convert(bn5_running_mean,param_lenth)
        bn5_running_var = param_convert(bn5_running_var,param_lenth)

        conv6_weight = param_convert(conv6_weight,param_lenth)
        conv6_bias = param_convert(conv6_bias,param_lenth)

        pos_weight = param_convert(pos_weight,param_lenth)
        pos_bias = param_convert(pos_bias,param_lenth)
        cos_weight = param_convert(cos_weight,param_lenth)
        cos_bias = param_convert(cos_bias,param_lenth)
        sin_weight = param_convert(sin_weight,param_lenth)
        sin_bias = param_convert(sin_bias,param_lenth)
        width_weight = param_convert(width_weight,param_lenth)
        width_bias = param_convert(width_bias,param_lenth)

        if args.jacquard_output:
            jo_fn = network + '_jacquard_output.txt'
            with open(jo_fn, 'w') as f:
                pass

        start_time = time.time()
        results = {'correct': 0, 'failed': 0}

        for idx, (x, y, didx, rot, zoom) in enumerate(test_data):

            conv1_out = F.conv2d(x,conv1_weight,bias=conv1_bias,padding=4)
            bn1_out = F.batch_norm(conv1_out, bn1_running_mean, bn1_running_var, weight=bn1_weight, bias=bn1_bias)
            relu_1 = F.relu(bn1_out)

            conv2_out = F.conv2d(relu_1,conv2_weight,bias=conv2_bias,padding=1,stride=2)
            bn2_out = F.batch_norm(conv2_out, bn2_running_mean, bn2_running_var, weight=bn2_weight, bias=bn2_bias)
            relu_2 = F.relu(bn2_out)

            conv3_out = F.conv2d(relu_2,conv3_weight,bias=conv3_bias,padding=1,stride=2)
            bn3_out = F.batch_norm(conv3_out, bn3_running_mean, bn3_running_var, weight=bn3_weight, bias=bn3_bias)
            relu_3 = F.relu(bn3_out)

            res1_conv1_out = F.conv2d(relu_3,res1_conv1_weight,bias=res1_conv1_bias,padding=1)
            res1_bn1_out   = F.batch_norm(res1_conv1_out,res1_bn1_running_mean,res1_bn1_running_var,weight=res1_bn1_weight,bias=res1_bn1_bias)
            relu_res1      = F.relu(res1_bn1_out)
            res1_conv2_out = F.conv2d(relu_res1,res1_conv2_weight,bias=res1_conv2_bias,padding=1)
            res1_bn2_out   = F.batch_norm(res1_conv2_out,res1_bn2_running_mean,res1_bn2_running_var,weight=res1_bn2_weight,bias=res1_bn2_bias)

            res1_out = relu_3 + res1_bn2_out

            res2_conv1_out = F.conv2d(res1_out,res2_conv1_weight,bias=res2_conv1_bias,padding=1)
            res2_bn1_out   = F.batch_norm(res2_conv1_out,res2_bn1_running_mean,res2_bn1_running_var,weight=res2_bn1_weight,bias=res2_bn1_bias)
            relu_res2      = F.relu(res2_bn1_out)
            res2_conv2_out = F.conv2d(relu_res2,res2_conv2_weight,bias=res2_conv2_bias,padding=1)
            res2_bn2_out   = F.batch_norm(res2_conv2_out,res2_bn2_running_mean,res2_bn2_running_var,weight=res2_bn2_weight,bias=res2_bn2_bias)

            res2_out = res1_out + res2_bn2_out

            res3_conv1_out = F.conv2d(res2_out,res3_conv1_weight,bias=res3_conv1_bias,padding=1)
            res3_bn1_out   = F.batch_norm(res3_conv1_out,res3_bn1_running_mean,res3_bn1_running_var,weight=res3_bn1_weight,bias=res3_bn1_bias)
            relu_res3      = F.relu(res3_bn1_out)
            res3_conv2_out = F.conv2d(relu_res3,res3_conv2_weight,bias=res3_conv2_bias,padding=1)
            res3_bn2_out   = F.batch_norm(res3_conv2_out,res3_bn2_running_mean,res3_bn2_running_var,weight=res3_bn2_weight,bias=res3_bn2_bias)

            res3_out = res2_out + res3_bn2_out

            res4_conv1_out = F.conv2d(res3_out,res4_conv1_weight,bias=res4_conv1_bias,padding=1)
            res4_bn1_out   = F.batch_norm(res4_conv1_out,res4_bn1_running_mean,res4_bn1_running_var,weight=res4_bn1_weight,bias=res4_bn1_bias)
            relu_res4      = F.relu(res4_bn1_out)
            res4_conv2_out = F.conv2d(relu_res4,res4_conv2_weight,bias=res4_conv2_bias,padding=1)
            res4_bn2_out   = F.batch_norm(res4_conv2_out,res4_bn2_running_mean,res4_bn2_running_var,weight=res4_bn2_weight,bias=res4_bn2_bias)

            res4_out = res3_out + res4_bn2_out

            res5_conv1_out = F.conv2d(res4_out,res5_conv1_weight,bias=res5_conv1_bias,padding=1)
            res5_bn1_out   = F.batch_norm(res5_conv1_out,res5_bn1_running_mean,res5_bn1_running_var,weight=res5_bn1_weight,bias=res5_bn1_bias)
            relu_res5      = F.relu(res5_bn1_out)
            res5_conv2_out = F.conv2d(relu_res5,res5_conv2_weight,bias=res5_conv2_bias,padding=1)
            res5_bn2_out   = F.batch_norm(res5_conv2_out,res5_bn2_running_mean,res5_bn2_running_var,weight=res5_bn2_weight,bias=res5_bn2_bias)

            res5_out = res4_out + res5_bn2_out

            conv4_out = F.conv_transpose2d(res5_out,conv4_weight,bias=conv4_bias,padding=1,output_padding=1,stride=2)
            bn4_out = F.batch_norm(conv4_out,bn4_running_mean,bn4_running_var,weight=bn4_weight,bias=bn4_bias)
            relu_4 = F.relu(bn4_out)

            conv5_out = F.conv_transpose2d(relu_4,conv5_weight,bias=conv5_bias,padding=2,output_padding=1,stride=2)
            bn5_out = F.batch_norm(conv5_out,bn5_running_mean,bn5_running_var,weight=bn5_weight,bias=bn5_bias)
            relu_5 = F.relu(bn5_out)

            conv6_out = F.conv_transpose2d(relu_5,conv6_weight,bias=conv6_bias,padding=4,stride=1)

            pos_out = F.conv2d(conv6_out,pos_weight,bias=pos_bias)
            cos_out = F.conv2d(conv6_out,cos_weight,bias=cos_bias)
            sin_out = F.conv2d(conv6_out,sin_weight,bias=sin_bias)
            width_out = F.conv2d(conv6_out,width_weight,bias=width_bias)

            loss_fix = loss(pos_out,cos_out,sin_out,width_out,y)

            # loss_origin = net.compute_loss(test_features, test_labels)
            # bn1_out = bn1_out.numpy()
            # bn1_out_fix = Fxp(bn1_out, True, 8, 2, overflow='saturate')
            # bn1_out_fix_numpy = bn1_out_fix.get_val()
            # bn1_final = torch.from_numpy(bn1_out_fix_numpy)
            # for i in range(len(bn1_out[0])):
            #     for j in range(len(bn1_out[0][0])):
            #         for k in range(len(bn1_out[0][0][0])):
            #             bn1_out[0][i][j][k] = ((conv1_out[0][i][j][k]-bn1_running_mean[i])/math.sqrt(bn1_running_var[i]+0.00001))*bn1_weight[i] + bn1_bias[i]
                        # bn1_out[0][i][j][k] = ((conv1_out[0][i][j][k]-bn1_running_mean[i])/bn1_running_var[i])
            # relu_1 = torch.nn.functional.relu(bn1_out)
            # print("size of fixed data = ",loss_fix['pred']['pos'].size())
            # print("size of x_test = ",loss_origin['pred']['pos'].size())
            # print("floating data = ",loss_fix['pred']['pos'])
            # print(loss_origin['pred']['pos'])
            

            q_img, ang_img, width_img = post_process_output(loss_fix['pred']['pos'], loss_fix['pred']['cos'],
                                                                    loss_fix['pred']['sin'], loss_fix['pred']['width'])

            if args.iou_eval:
                s = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom),
                                                no_grasps=args.n_grasps,
                                                grasp_width=width_img,
                                                threshold=args.iou_threshold
                                                )
                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1


        
        # with torch.no_grad():
        #     for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
        #         if idx==1 : pass
        #         print(x.size())
        #         x = x.numpy()
        #         print(np.amax(x))
        #         print(type(x[0][0][0][0]))
        #         print(x[0][0][0][0])
        #         xc = x.to(device)
        #         yc = [yi.to(device) for yi in y]
        #         # print("xc, yc = ",xc,yc)

        #         lossd = net.compute_loss(xc, yc)

        #         q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
        #                                                         lossd['pred']['sin'], lossd['pred']['width'])

        #         if args.iou_eval:
        #             s = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom),
        #                                                no_grasps=args.n_grasps,
        #                                                grasp_width=width_img,
        #                                                threshold=args.iou_threshold
        #                                                )
        #             if s:
        #                 results['correct'] += 1
        #             else:
        #                 results['failed'] += 1

        #         if args.jacquard_output:
        #             grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
        #             with open(jo_fn, 'a') as f:
        #                 for g in grasps:
        #                     f.write(test_data.dataset.get_jname(didx) + '\n')
        #                     f.write(g.to_jacquard(scale=1024 / 300) + '\n')

        #         if args.vis:
        #             save_results(
        #                 rgb_img=test_data.dataset.get_rgb(didx, rot, zoom, normalise=False),
        #                 depth_img=test_data.dataset.get_depth(didx, rot, zoom),
        #                 grasp_q_img=q_img,
        #                 grasp_angle_img=ang_img,
        #                 no_grasps=args.n_grasps,
        #                 grasp_width_img=width_img
        #             )

        avg_time = (time.time() - start_time) / len(test_data)
        logging.info('Average evaluation time per image: {}ms'.format(avg_time * 1000))

        if args.iou_eval:
            logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                                                      results['correct'] + results['failed'],
                                                      results['correct'] / (results['correct'] + results['failed'])))

        if args.jacquard_output:
            logging.info('Jacquard output saved to {}'.format(jo_fn))

        del net
        torch.cuda.empty_cache()
