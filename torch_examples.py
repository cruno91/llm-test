import torch

# Check if Metal is available.
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")

# Example datasets of tensors.
print("Example datasets of tensors.")
print("Random integers from -100, to 100, with a shape of 6.")
randint = torch.randint(-100, 100, (6,))
print(randint)
print(" ")
print("------------")
print(" ")

print("Tensor")
tensor = torch.tensor([[0.1, 0.2], [2.2, 3.1], [4.9, 5.2]])
print(tensor)
print(" ")
print("------------")
print(" ")

print("Tensor with zeroes with a shape of 2x3.")
zeros = torch.zeros(2, 3)
print(zeros)
print(" ")
print("------------")
print(" ")

print("Tensor with ones with a shape of 3x4.")
ones = torch.ones(3, 4)
print(ones)
print(" ")
print("------------")
print(" ")

print("Tensor with zero values with a shape of 2x3.")
input = torch.empty(2, 3)
print(input)
print(" ")
print("------------")
print(" ")

print("Tensor with 5 integers from 0 to 4.")
arange = torch.arange(5)
print(arange)
print(" ")
print("------------")
print(" ")

# Increments of the steps from 3 to 10 in 5 steps.
print("Increments of the steps from 3 to 10 in 5 steps.")
linespace = torch.linspace(3, 10, steps=5)
print(linespace)
logspace = torch.logspace(-10, 10, steps=5)
print(logspace)
print(" ")
print("------------")
print(" ")

# 5x5 matrix with a diagonal of ones.
print("5x5 matrix with a diagonal of ones.")
eye = torch.eye(5)
print(eye)
a = torch.empty((2, 3), dtype=torch.int64)
empty = torch.empty_like(a)
print(empty)
print(" ")
print("------------")
print(" ")

# Probablity distributions.
print("Probablity distributions.")
prob = torch.tensor([0.1, 0.9])
# 10% or 90%, each probability points to the index of the probability in the tensor.
# Draw 5 samples from the multinomial distribution.
samples = torch.multinomial(prob, num_samples=10, replacement=True)
print(samples)
print(" ")
print("------------")
print(" ")

# Concat tensors.
print("Concat tensors.")
# This is used for the output of the predictions.
print("This is used for the output of the predictions.")
a = torch.tensor([1, 2, 3, 4])
out = torch.cat((a, torch.tensor([5])), dim=0)
print(out)
print(" ")
print("------------")
print(" ")

# Tril = triangle lower
print("Triangle lower")
# As each row is processed, there is more history to look back on.
print("As each row is processed, there is more history to look back on.")
# You have to know based on your history to know what the next word is.
print("You have to know based on your history to know what the next word is.")
out = torch.tril(torch.ones(5, 5))
print(out)
print(" ")
print("------------")
print(" ")

# Triu = triangle upper
print("Triangle upper")
out = torch.triu(torch.ones(5, 5))
print(out)
print(" ")
print("------------")
print(" ")

# Masked fill
print("Masked fill")
# In order to get to tril, exponentiate every element in the matrix.
print("In order to get to tril, exponentiate every element in the matrix.")
out = torch.zeros(5, 5).masked_fill(torch.tril(torch.ones(5, 5)) == 0 , float('-inf'))
print(out)
print(" ")
print("------------")
print(" ")

# Transposing
print("Transposing")
# Swap the dimensions of a tensor.
print("Swap the dimensions of a tensor.")
# 2x3x4 tensor.
print("2x3x4 tensor.")
input = torch.zeros(2, 3, 4)
print(input)
# 4x3x2 tensor.
print("Transposed version of the tensor (4x3x2).")
out = input.transpose(0, 2)
print(out)
print(" ")
print("------------")
print(" ")

# Stacks tensors along a new dimension.
print("Stack tensors along a new dimension.")
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
tensor3 = torch.tensor([7, 8, 9])
# Use this to stack blocks to make a batch.
stacked_tensor = torch.stack([tensor1, tensor2, tensor3])
print(stacked_tensor)

# nn.linear function
print("nn.linear function")
import torch.nn as nn
sample = torch.tensor([10., 10., 10.])
linear = nn.Linear(3, 3, bias=False)
print(linear(sample))

# 57:27