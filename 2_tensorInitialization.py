import torch
import numpy as np

#================================================================================================#
#                                    Initializing tensor                                         #
#================================================================================================#

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3], [5,6,7], [9, 10, 11]], dtype=torch.float32,
                         device=device, requires_grad=True )
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

#Orther common initialization method
x = torch.empty(size=(3,3))
x = torch.zeros(size=(3,3))
x = torch.rand(size=(3,3))
x = torch.ones(size=(3,3))
x = torch.eye(5,5)
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0.1, end=1, steps=10)
x = torch.empty(size=(1,5)).normal_(mean=0, std=1)
x = torch.empty(size=(1,5)).uniform_(0, 1)
x = torch.diag(torch.ones(3))

#How to initialize and convert to other type (float, int, double)
tensor = torch.arange(5)
print(tensor)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())
print(tensor.double())

#Array to tensor conversion and vice versa
np_array = np.zeros((3,3))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy( )