import torch
import sys
from fxpmath import Fxp
import numpy as np

x = Fxp(-5.25789021)
print (x.info())



# network = sys.argv[1]
network = "trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98"

net = torch.load(network)
# net_q = torch.quantization.quantize_dynamic(net, qconfig_spec=None, dtype=torch.qint8, mapping=None, inplace=False)
# net.eval()
model_dict = net.state_dict()
# model_dict_q = net_q.state_dict()
# print ("model_dict = ",model_dict)
keys = model_dict.keys()
print ("keys = ",keys)

conv1_weight = model_dict['conv1.weight']

# conv1_weigdt_fixed = model_dict_q['conv1.weight']

# print("origin = ",conv1_weight[0][0][0])
# print("quantize = ",conv1_weigdt_fixed[0][0][0])
# print ("conv1_weight = ",conv1_weight.size())
# bn1_weight = model_dict["bn1.weight"]
# print ("bn1_weight = ",bn1_weight)

for key in model_dict:
	message = "Layer: {} Size: {}".format(key,model_dict[key].size())
	print (message)
	x = model_dict[key].numpy()
	print(np.amax(x))
# print ("'conv1.bias' = ",model_dict["'conv1.bias'"])
# print ("Net = ",net)
# print ("Type of net = ", type(net))
# weights = net.weights()
# print (weights)
# print (net.conv1)

