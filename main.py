import torch

# Check if Metal is available.
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

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