import torch

x = torch.rand(2,3, dtype=torch.float32) 


# x
# tensor([[0.6839, 0.4741, 0.7451],
#         [0.9301, 0.1742, 0.6835]])

print (x)

xq = torch.quantize_per_tensor(x, scale = 0.5, zero_point = 8, dtype=torch.quint8)
# tensor([[0.5000, 0.5000, 0.5000],
#         [1.0000, 0.0000, 0.5000]], size=(2, 3), dtype=torch.quint8,
#        quantization_scheme=torch.per_tensor_affine, scale=0.5, zero_point=8)

print (xq)

print (xq.int_repr())
# tensor([[ 9,  9,  9],
#         [10,  8,  9]], dtype=torch.uint8)

