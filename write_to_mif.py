import argparse
import torch
import torch.utils.data
import torch.nn.functional as F
from utils.data import get_dataset
import memory_init_utils
import numpy
# from utils.dataset_processing import evaluation, grasp
# from utils.visualisation.plot import save_results
from inference_fxp import mif_weight_convert, loss, calculate_max, param_convert, feature_convert

network_path = "trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98"


def parse_args():
	parser = argparse.ArgumentParser(description='Evaluate networks')

	# Network
	parser.add_argument('--network', metavar='N', type=str, nargs='+',
						help='Path to saved networks to evaluate')



if __name__ == '__main__':
	args = parse_args()
	net = torch.load(network_path, map_location="cpu")
	model_dict = net.state_dict()
	conv1_weight = model_dict['conv1.weight'].numpy()
	weight_length = 11
	conv1_weight_converted = mif_weight_convert(conv1_weight,weight_length)
	conv1_weight_param= param_convert(conv1_weight,weight_length)

	print ("original = ",conv1_weight[0][0][0][0])
	print ("param_convert = ",conv1_weight_param[0][0][0][0])
	print ("mif_convert = ",conv1_weight_converted[0][0][0][0])
	# print ("conv1_weight = ",conv1_weight)
	test_data = numpy.array([[0,1],
							 [1,2],
							 [2,3],
							 [3,4],
							 [4,5]],dtype=numpy.uint8)

	with open("test.mif","w") as write_file:
		result = memory_init_utils.dumps(test_data,write_file)
		print (result)

