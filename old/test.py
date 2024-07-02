import torch

from nectar.utils.mi import mutual_information


x = torch.randn(5, 3)  # Input tensor
y = torch.randint(low=1, high=3, size=(5,))  # Target tensor

print(y)
# one hot encode y
y = torch.nn.functional.one_hot(y, num_classes=3).float()
print(y)

print(x.shape, x.dtype)
print(y.shape, y.dtype)

mi = mutual_information(x, y, dist_type="gaussian")
print(mi)
