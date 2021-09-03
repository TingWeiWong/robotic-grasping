import argparse
import torch
import torch.utils.data
import torch.nn.functional as F
from utils.data import get_dataset
# from utils.dataset_processing import evaluation, grasp
# from utils.visualisation.plot import save_results

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
	print ("conv1_weight = ",conv1_weight)

