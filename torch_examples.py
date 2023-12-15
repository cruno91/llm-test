import torch

# Check if Metal is available.
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

# Example datasets of tensors.
randint = torch.randint(-100, 100, (6,))
print(randint)
tensor = torch.tensor([[0.1, 0.2], [2.2, 3.1], [4.9, 5.2]])
print(tensor)
zeros = torch.zeros(2, 3)
print(zeros)
ones = torch.ones(3, 4)
print(ones)
input = torch.empty(2, 3)
print(input)
arange = torch.arange(5)
print(arange)
# Increments of the steps from 3 to 10 in 5 steps.
linespace = torch.linspace(3, 10, steps=5)
print(linespace)
logspace = torch.logspace(-10, 10, steps=5)
print(logspace)
# 5x5 matrix with a diagonal of ones.
eye = torch.eye(5)
print(eye)
a = torch.empty((2, 3), dtype=torch.int64)
empty = torch.empty_like(a)
print(empty)