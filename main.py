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
