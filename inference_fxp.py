import torch
import sys
import numpy as np
from fxpmath import Fxp
import torch.nn.functional as F

network = "trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98"
net = torch.load(network)

model_dict = net.state_dict()
keys = model_dict.keys()
# print ("keys = ",keys)

conv1_weight = model_dict['bn2.num_batches_tracked']
# print(conv1_weight)

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