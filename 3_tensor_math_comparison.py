import torch

#================================================================================================#
#                      tensor maths and comparison operations                                    #
#================================================================================================#

x = torch.tensor([1, 2, 3])
y = torch.tensor([5, 6, 7])

#Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)

z2 = torch.add(x, y)
z = x + y

#substraction
z = x - y

#Division
z = torch.true_divide(x, y)

#inplace operations
t = torch.zeros(3)
t.add_(x)
t += x

#exponentiation
z = x.pow(2)
z = x**2

#simple comparison
z = x > 0
z = x < 0
z = x >= 0
z = x <= 0
z = x == 0
z = torch.eq(x, y)
z = torch.ne(x, y)
z = torch.lt(x, y)
z = torch.gt(x, y)
z = torch.le(x, y)
z = torch.ge(x, y)

#Matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) # (2, 5) * (5, 2) = (2, 3)
x3 = x1.mm(x2)

#matrix exponentiation
matrix_exp = torch.rand((5, 5))
print(matrix_exp.matrix_power(3))

#element wise multiplication
z = x * y
print(z)

#dot product
z = torch.dot(x, y)
z = torch.matmul(x, y)
print(z)

#Batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2) # (batch, n, p)
print(out_bmm)

#Orther useful tensor operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(values,dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0, max=10)




