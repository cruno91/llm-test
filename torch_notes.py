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
print(" ")
print("------------")
print(" ")

# nn.linear function
print("nn.linear function")
import torch.nn as nn
sample = torch.tensor([10., 10., 10.])
print(sample)
# Linear transformation.
linear = nn.Linear(3, 3, bias=False)
print(linear(sample))
# When apply weight or bias under nn.module it will learn those
# and become better and train based on how accurate those are
# or how close parameters bring it to desired output.
# Docs > torch.nn
print(" ")
print("------------")
print(" ")

# Softmax function
# Convert a tensor of numbers into a tensor of probabilities.
print("Softmax function")
# [1.0, 2.0, 3.0] -> [x, y, z]
# Exponentiate each of those numbers.
# (1).exp() = e^1 = 2.718281828459045
# (2).exp() = e^2 = 7.38905609893065
# (3).exp() = e^3 = 20.085536923187668
# Add them up.
# 2.718281828459045 + 7.38905609893065 + 20.085536923187668 = 30.19287485057736
# Divide each number by the sum.
# 2.718281828459045 / 30.19287485057736 = 0.09003057317038046
# 7.38905609893065 / 30.19287485057736 = 0.24472847105479764
# 20.085536923187668 / 30.19287485057736 = 0.6652409557748219
# The sum of the probabilities should be 1.
# 0.09003057317038046 + 0.24472847105479764 + 0.6652409557748219 = 1.0
# The softmax function is used to convert a tensor of numbers into a tensor of probabilities.
import torch.nn.functional as F
# Create a tensor.
tensor1 = torch.tensor([1.0, 2.0, 3.0])
# Apply softmax function.
# dim=0: sum of the probabilities should be 1. - 1 dimension, a line
softmax_output = F.softmax(tensor1, dim=0)
print(softmax_output)
print(" ")
print("------------")
print(" ")

# Embedding vectors
# Stores some information about a word or *character*, like from a vocabulary.
print("Embedding vectors")
# A vector or numerical representation of a character.
# Initialize the embedding layer.
voacb_size = 1000
embedding_dim = 100
embedding = nn.Embedding(voacb_size, embedding_dim)
# Create some input indicies.
input_indicies = torch.LongTensor([1, 5, 3, 2])
# Apply the embedding layer.
embedding_output = embedding(input_indicies)
# The output will be a tensor of shape (4, 100), where 4 is the number of inputs
# and 100 is the dimensionality of the embedding vectors.
print(embedding_output.shape)
print(" ")
print("------------")
print(" ")

# Matrix multiplication.
# The dot product of two vectors.
# Given two vectors:
# [1, 2, 3]
# [4, 5, 6]
# Multiply the corresponding elements and add them up to get the dot product.
# 1 * 4 + 2 * 5 + 3 * 6 = 32
# The 2 matrix need to be able to be multiplied.
# A 3x2 and a 2x3 can be multiplied.
# A 3x4 and a 5x1 cannot be multiplied.
# The number of columns in the first matrix must match the number of rows in the second matrix.
# [
#   1, 2
#   3, 4
#   5, 6
# ]
# [
#   2,  8,  9
#   10, 11, 12
# ]
# (1x7)+(2x10) = 27
# (1x8)+(2x11) = 30
# (1x9)+(2x12) = 33
# (3x7)+(4x10) = 61
# (3x8)+(4x11) = 68
# (3x9)+(4x12) = 75
# (5x7)+(6x10) = 95
# (5x8)+(6x11) = 106
# (5x9)+(6x12) = 117
print("Matrix multiplication")
a = torch.tensor([[1, 2], [3, 4], [5, 6]])
b = torch.tensor([[2, 8, 9], [10, 11, 12]])
# The @ symbol is the matrix multiplication operator in pytorch.
# You can also do print(torch.matmul(a, b))
print(a @ b)
print(" ")
print("------------")
print(" ")

# In PyTorch you cannot multiply integers and floats.
print("In PyTorch you cannot multiply integers and floats.")
int_64 = torch.randint(1, (3, 2))
float_32 = torch.rand(2,3)
# result = torch.matmul(int_64, float_32)
# Above will fail.
# You can change the type of the tensor.
int_64 = torch.randint(1, (3, 2)).float()
result = torch.matmul(int_64, float_32)
print(result)
print(" ")
print("------------")
print(" ")

# Gradient descent.
# The gradient is the slope of the loss function.
# The loss function is the difference between the predicted value
# and the actual value.
# ...or the mean squared error.
# You need to pass nn.Module to the optimizer (class).
# Example: If you have 80 characters in vocab and have just
# started model with no training and random weights.
# There is a 1 in 80 chance the next token is predicted successfully.
# How to measure the loss is the negative log likelihood ( -ln(1/80) )
# (not even 2% chance).
# You want to minimize loss and increase prediction accuracy.
# Take the derivative of the current point of where it's at now
# and move it in a different direction.
# Picture a slope with a line going across the top of it which hits
# the slope at a point.  The line is the loss function.
# Take the derivative of the current point and move it in a different direction.
# Gradient descent is an optimizer.
# The point should be moved to where the derivative is heading down
# the hill.
# torch.optim is a package implementing various optimization algorithms.

# Learning rate
# Say you decide you need to take a big step in the direction of
# the derivative (Gradient descent).
# You might overshoot the minimum (bottom of the slope).
# You need to take smaller steps so the parameters don't change too
# much.

# AdamW optimizer
# AdamW takes a generalized form of gradient descent and adds
# a momentum term (weight decay).
# The weight significance shrinks as gradient descent flattens out
# so that certain weights don't become too large.

# logits.view
print ("logits.view")
a = torch.rand(2, 3, 5)
print(a.shape)
# Unbpack the shape.
x, y, z = a.shape
# Reshape as tensor.
a = a.view(x, y, z)
print(x, y, z)
print(a.shape)
print(" ")
print("------------")
print(" ")

# Optimizers:
# Mean Squared Error (MSE): A common loss function used in regression
# problems, where the goal is to predict a continous output. It measures
# the average squared difference between the predicted and actual values,
# and is often used to train neural networks for regression tasks.

# Gradient Descent (GD): An optimization algorithim used to minimize
# the loss function of a machine learning model.  The loss function
# measures how well the model is able to predict the target variable
# based on the input features. The idea of GD is to iteratively
# adjust the model parameters in the direction of the steepest descent
# of the loss function.

# Momentium: An extension of SGD that adds a "momentum" term to the
# parameter updates.  This term helps smooth out the updates and allows
# the optimizer to continue moving in the right direction, even if the
# gradient changes direction or varies in magnitude. Useful for training
# deep neural networks.

# RMSprop: Uses a moving average of squared gradients to normalize the
# rate of each parameter.  Helps avoid oscillations in the parameter updates
# and can improve the rate of convergence.

# Adam: Uses moving average of both the gradient and its squared value to
# adapt the learning rate of each parameter. Often  used as default for
# deep learning models
# Combines the advantages of RMSprop and momentum.

# AdamW: Modification of Adam that adds weight decay to the parameter
# updates. Helps to regularize the model and can improve generaization
# performance.
