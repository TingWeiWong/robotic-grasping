import torch
import sys
import numpy as np
from fxpmath import Fxp
import torch.nn.functional as F

def mif_weight_convert(weight_matrix, bitwidth):
	"""
	This function reduces bit precision of the 
	weight matrix according to bidwidth
	- Input:
		* weight_matrix: conv, batch_norm, fully_connected weights 
			ex. [[0.1,0.2,0.3],[0.1,-0.2,0.3],..]
		* bitwidth: length of max binary representation ex. 5
	- Output:
		* bin_weight: Flattened binary quantized version of weight_matrix 
	"""
	Flattened_array = weight_matrix.flatten()
	print ("Flattened_array = ",Flattened_array[0:10])
	for address, value in enumerate(Flattened_array):
		# Should detect overflow
		quantized_weight = Fxp(value, signed=True, n_word=bitwidth+1, n_frac=bitwidth, overflow='saturate', rounding='around').bin(frac_dot=True)
		# print ("quantized_weight = ",quantized_weight)

	# numpy_quantized_weight = quantized_weight.bin()
	# print ("numpy_quantized_weight = ",type(numpy_quantized_weight))

	# numpy_bin_weight = np.binary_repr(quantized_weight.get_val(),width=bitwidth)
	# print ("bin_weight = ",bin_weight[0][0][0][0])
	# bin_weight = bin_weight.get_val()	
	# print ("bin_weight = ",bin_weight[0][0][0][0])	
	# bin_weight = torch.from_numpy(bin_weight)
	# print ("bin_weight = ",bin_weight[0][0][0][0])
	# bin_weight = bin_weight.type(torch.FloatTensor)


	return numpy_bin_weight

def param_convert(x, a):
	y = Fxp(x, signed=True, n_word=a+1, n_frac=a, overflow='saturate', rounding='around')
	y = y.get_val()
	if(a==100):
		y = torch.from_numpy(x)
	else:
		max = 0
		y_a  = np.abs(y)
		
		# for i in range(len(y)):
		#     s = y_a[i]
		#     if(s>=max):
		#         max = s
		# print("max = ", s)
		y = torch.from_numpy(y)
	y = y.type(torch.FloatTensor)
	return y

def feature_convert(x, integer, decimal):
	y = x.numpy()
	# y = Fxp(y, signed=True, n_word=integer+decimal+1, n_frac=decimal, overflow='saturate', rounding='around')
	# y = y.get_val()
	if(decimal==100):
		y = torch.from_numpy(y)
	else:
		y = Fxp(y, signed=True, n_word=integer+decimal+1, n_frac=decimal, overflow='saturate', rounding='around')
		y = y.get_val()
		y = torch.from_numpy(y)
	y = y.type(torch.FloatTensor)
	return y

def loss(pos_pred, cos_pred, sin_pred, width_pred, yc):
		y_pos, y_cos, y_sin, y_width = yc

		p_loss = F.smooth_l1_loss(pos_pred, y_pos)
		cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
		sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
		width_loss = F.smooth_l1_loss(width_pred, y_width)

		return {
			'loss': p_loss + cos_loss + sin_loss + width_loss,
			'losses': {
				'p_loss': p_loss,
				'cos_loss': cos_loss,
				'sin_loss': sin_loss,
				'width_loss': width_loss
			},
			'pred': {
				'pos': pos_pred,
				'cos': cos_pred,
				'sin': sin_pred,
				'width': width_pred
			}
		}

def calculate_max(x, max):
	x = x.numpy()
	if(np.amax(x)>=max):
		max_out = np.amax(x)
	else:
		max_out = max
	return max_out





if __name__ == "__main__":
	network = "trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98"
	net = torch.load(network)

	model_dict = net.state_dict()
	keys = model_dict.keys()
	# print ("keys = ",keys)

	conv1_weight = model_dict['bn2.num_batches_tracked']
	# print(conv1_weight)	