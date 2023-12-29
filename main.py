import torch
import torch.nn as nn
import torch.nn.functional as F

# Check if Metal is available.
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print(x)
else:
    device = torch.device("cpu")
    print("MPS device not found.")

# Open the text file.
with open('wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# Get the set of unique characters in the text.
chars = sorted(set(text))
print(chars)

# Create a tokenizer to convert between characters and numerical indices via an encoder and a decoder.
# Encoder and a decoder:
# Encoder: converts a string to an integer.
strings_to_ints = {c: i for i, c in enumerate(chars)}
encode = lambda s: [strings_to_ints[c] for c in s]
# Decoder: converts an integer to a string.
ints_to_strings = {i: c for i, c in enumerate(chars)}
decode = lambda x: ''.join([ints_to_strings[i] for i in x])

# Convert the text to integers.
# encoded = encode("Hello")
# print(encoded)
# decoded = decode(encoded)
# print(decoded)

# Convert the text to integers.
# dtype = torch.long: 64-bit integer (signed) - Important for pytorch to know the type of data.
data = torch.tensor(encode(text), dtype=torch.long)

# You have to split your training data corpus into chunks so that you create output like the original, but not copies
# of the original.
# The chunks are also needed to be able to scale the model.
# It's called the bi-gram language model.
block_size = 8
# Number of blocks you can do in parallel.
batch_size = 4

# Get the training and validation splits.
n = int(0.8*len(data))
train_data, val_data = data[:n], data[n:]

# Get a batch of data.
def get_batch(split):
    data = train_data if split == "train" else val_data
    # Take a random integer between 0 and the length of the data minus the block size.
    # So if you get the index that's at the length of the n minus block size you'll still have a block size of 8.
    ix = torch.randint(len(data) - block_size, (batch_size,))
    print(ix)
    # Get the data from the random index to the random index plus the block size.
    x = torch.stack([data[i:i+block_size] for i in ix])
    # Get the data from the random index plus 1 to the random index plus the block size plus 1.
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # Move the data to the device.
    x, y = x.to(device), y.to(device)
    return x, y

x, y = get_batch("train")
print('inputs:')
print(x)
print('targets:')
print(y)

# Bigram language model.
# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print("when input is", context, "target is", target)

# Create the model.
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Embedding layer: Converts an integer to a vector.
        # Lookup table tokens in a block.  Each token is a number with a probability distribution
        # across the vocabulary to predict the next token.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, index, targets=None):
        # Logits are the output of the model before the softmax activation function.
        # The logits are the raw values that are passed to the activation function.
        # The activation function then normalizes the logits and converts them to a probability distribution.
        # Logits are the unnormalized output of a neural network.
        # Layman: Logits are a bunch of normalized floating point numbers.
        # Say you have [2, 4, 6] and you want to normalize it.
        # You would divide each number by the sum of the numbers.
        # 2 / (2 + 4 + 6) = 0.1
        # 4 / (2 + 4 + 6) = 0.2
        # 6 / (2 + 4 + 6) = 0.3
        # The sum of the numbers is 1.
        # This becomes [0.1, 0.2, 0.3].
        # Say those equate to the probabilities of a, b, and c.
        # The probability of a is 0.1.
        # The probability of b is 0.2.
        # The probability of c is 0.3.
        logits = self.token_embedding_table(index)

        # Because targets is none, the loss is none and the code does not execute.
        # Just use the logits which are three-dimensional.
        if targets is None:
            loss = None
        else:
            # batch, time
            # (sequence of integers, we don't know next token, because some we don't know yet),
            # channels (vocab size)
            # Shape is BxTxC.
            B, T, C = logits.shape
            # Because we're paying attention to the vocabulary (channels), the batch
            # and time dimensions are combined.
            # As long as the logits and the targets are the same batch and time we should be alright.
            # PyTorch expects the logits to be a 2D tensor and the targets to be a 1D tensor which is why we use
            # view() to reshape the tensors.
            # Making the first parameter a single parameter of batch by time.
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # cross_entropy is way of measuring loss.
            loss = F.cross_entropy(logits, targets)
        return logits, loss
