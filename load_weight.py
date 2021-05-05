import torch
import sys
from fxpmath import Fxp

x = Fxp(-5.25)
print (x.info())


# network = sys.argv[1]
network = "trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98"
net = torch.load(network)

# net.eval()
model_dict = net.state_dict()
# print ("model_dict = ",model_dict)
keys = model_dict.keys()
print ("keys = ",keys)

conv1_weight = model_dict['conv1.weight']

print ("conv1_weight = ",conv1_weight.size())
bn1_weight = model_dict["bn1.weight"]
print ("bn1_weight = ",bn1_weight)

for key in model_dict:
	message = "Layer: {} Size: {}".format(key,model_dict[key].size())
	print (message)
# print ("'conv1.bias' = ",model_dict["'conv1.bias'"])
# print ("Net = ",net)
# print ("Type of net = ", type(net))
# weights = net.weights()
# print (weights)
# print (net.conv1)

