import torch
import sys
from fxpmath import Fxp

network = "../trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98"
net = torch.load(network)

model_dict = net.state_dict()
keys = model_dict.keys()
print ("keys = ",keys)

conv1_weight = model_dict['conv1.weight'].numpy()
