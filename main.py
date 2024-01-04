import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmap
import random
import pickle
import argparse

# Check if Metal is available.
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found.")
else:
    device = torch.device("cpu")
    print("MPS device not found.")

parser = argparse.ArgumentParser("Example GPT LLM.")
parser.add_argument("--batch-size", type=int, help="train the model with a specified batch size")
args = parser.parse_args()

# Hyperparameters.
# Affects memory.
batch_size = args.batch_size if args.batch_size is not None else 128  # Change for GPU. (4 test, 128 train)
block_size = 64  # Change for GPU. (v1 8 test, 64 train) - (v2 32 test, x train)
# Does not affect memory.
max_iterations = 3000  # Change for GPU. (v1 1000 test, 3000 train) - (v2 200 test, x train)
learning_rate = 3e-4  # 3e-3 = 0.003 - 3e-4, 1e-3, 1e-4
eval_iterations = 500  # Change for purpose. (v1 250 test, 500 train) - (v2 100 test, x train)
# Affect memory.
n_embed = 384  # Amount of neurons in the embedding layer.
n_head = 4  # Amount of heads (in parallel). (v1 4 for mps 8 for cuda) - (v2 1 test)
n_layer = 4  # Amount of layers (equal to number of decoder blocks). (v1 4 for mps 8 for cuda) - (v2 1 test)
# Does not affect memory.
dropout = 0.2  # Dropout rate. 20% of the neurons will be turned off.


# Open the text file.
with open('vocab.txt', 'r', encoding='utf-8') as file:
    text = file.read()
    # Get the set of unique characters in the text.
    chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a tokenizer to convert between characters and numerical indices via an encoder and a decoder.
strings_to_ints = {c: i for i, c in enumerate(chars)}
encode = lambda s: [strings_to_ints[c] for c in s]
ints_to_strings = {i: c for i, c in enumerate(chars)}
decode = lambda x: ''.join([ints_to_strings[i] for i in x])


# Memory map for using small snippets of text from a single file of any size.
def get_random_chunk(split):
    filename = "output_train.txt" if split == 'train' else "output_val.txt"
    # Opened in binary mode.
    with open(filename, 'rb') as f:
        # Memory map the file. (Chunks of the file are loaded into memory as needed.)
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading.
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size * batch_size)

            # Seek to the random position and read the block of text.
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)

            # Decode the block to a string, ignoring any invalid byte sequences.
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')

            # Train and test splits.
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data


# Get a batch of data.
def get_batch(split):
    # Get the data from the training or validation split.
    data = get_random_chunk(split)
    # Get a random index.
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Get the data from the random index to the random index plus the block size.
    x = torch.stack([data[i:i + block_size] for i in ix])
    # Get the data from the random index plus 1 to the random index plus the block size plus 1.
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    # Move the data to the device.
    x, y = x.to(device), y.to(device)
    return x, y


# Estimate the training loss.
@torch.no_grad()
def estimate_loss():
    # Create a dictionary to store the losses.
    out = {}
    # Set the model to evaluation mode.
    model.eval()
    for split in ["train", "val"]:
        # Create a tensor to store the losses.
        losses = torch.zeros(eval_iterations)
        # Get the losses.
        for k in range(eval_iterations):
            # Get the batch.
            x, y = get_batch(split)
            # Forward pass.
            logits, loss = model(x, y)
            # Store the loss.
            losses[k] = loss.item()
        # Get the mean of the losses.
        out[split] = losses.mean()
    # Set the model back to training mode.
    model.train()
    return out


# Define the head.
# Scaled dot production attention.
# One head of self-attention.
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # Query, key, and value.
        # Keys and queries dot product together.
        # Scaling by 1/sart (length of a row in the keys or queries matrix or tensor)
        # Transform n_embed to head size. (Linear transformation to 96 features instead of 384).
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # Register this no look ahead masking in the model state.
        # Instead of having to re-initialize for every head for every forward and
        # backward pass, just add it to the model state.
        # Prevent overhead computation of having to do this over and over again.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input size (batch, time, channels).
        # Output size (batch, time, head size).
        batch, time, channels = x.shape
        # Call linear transformation on input x.
        k = self.key(x)
        q = self.query(x)
        # Compute the attention scores (affinities).
        # Transposing the key matrix.  Flips second to last dimension with last dimension (time and head size).
        # (batch, time, head size) @ (batch, head size, time ->[dot product] (batch, time, time).
        # Scaling by 1/sqrt(length of a row in the keys or queries matrix or tensor).
        # (batch, time, time) * (1/sqrt(head size)).
        # Do this to prevent the dot product from getting too large. Think of trying to listen to a multiple
        # conversations at once.  You can't do it if some voices are too loud, drowning out the others.
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # Weights are attention scores.
        # Mask out the upper triangular part of the weights.
        # As the time step advances, the model can only see the past.
        # For each value that's 0, make it -infinity so that softmax will take these values and exponentiate normalize
        # them, which will turn -infinity to 0, sharpen the distribution, and make the model more confident.
        weights = weights.masked_fill(self.tril[:time, :time] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        # Perform the weighted aggregation of the values.
        v = self.value(x)
        out = weights @ v
        return out


# Define the multi-head attention.
# Multiple heads of attention in parallel.
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Module list: Heads in parallel for each head.
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Projection from the heads to the embedding size.
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        # Dropout.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the heads together along the last dimension.
        # Last dimension in this case is (batch, time, channels).
        # (batch, time, features) -> (batch, time, [head1, head 1, head 1, head1, head2 ... , head3 ...])
        # Above example is four features per head, and 3 heads.
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# Define the feed forward layer.
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),  # Rectified linear unit.  Converting values if equal to or below 0.
            nn.Linear(4 * n_embed, n_embed),  # Make sure Linear layers are equal to each other.
            nn.Dropout(dropout)  # Prevents over-fitting.
        )

    def forward(self, x):
        return self.net(x)


# Define the transformer block.
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        # The n_embed is the amount of neurons in the embedding layer.
        # AKA the amount of channels or heads we'd like.
        super().__init__()
        # Number of features each head will be capturing in multi-head attention.
        head_size = n_embed // n_head
        # Multi-head attention layer.
        # The sa is self-attention.
        self.sa = MultiHeadAttention(n_head, head_size)
        # Feedforward layer.
        self.ffwd = FeedForward(n_embed)
        # Layer normalization.
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    # Forward pass.
    def forward(self, x):
        # Self attention.
        y = self.sa(x)
        # Add and normalize rather than normalize and add.
        x = self.ln1(x + y)
        # Feed forward.
        y = self.ffwd(x)
        # Add and normalize again.
        x = self.ln2(x + y)
        return x


# Define the model.
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Embedding layer.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # Positional embedding layer.
        # Each letter index has a corresponding embedding vector.
        self.positioning_embedding_table = nn.Embedding(block_size, n_embed)
        # How many decoder layers to use simultaneously.
        # Asterisk is for repeating code in the brackets.
        # Will make 4 of the "Block"s (decoder layers) simultaneously.
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        # Final layer normalization.
        # Helps model converge better.
        self.ln_f = nn.LayerNorm(n_embed)
        # Linear layer for the language model header.
        self.lm_head = nn.Linear(n_embed, vocab_size)
        # Initialize the weights.
        self.apply(self._init_weights)

    # Initialize the weights around certain standard deviations.
    # Using 0.02 for the standard deviation is a common practice.
    # Helps training converge better.
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Forward pass.
    def forward(self, index, targets=None):
        # print(index.shape)
        batch, time = index.shape
        # index and targets are both (batch, time) tensors of integers.
        token_embeddings = self.token_embedding_table(index)  # (batch, time, channels)
        # Get the positional embeddings.
        positional_embeddings = self.positioning_embedding_table(torch.arange(time, device=device))  # (time, channels)
        # Use "x" to save memory?
        # Add the positional embeddings to the token embeddings.
        x = token_embeddings + positional_embeddings  # (embeddings) - (batch, time, channels)
        # Get the blocks.
        x = self.blocks(x)  # (blocks) - (batch, time, channels)
        # Get the final layer normalization.
        x = self.ln_f(x)  # (layer norm) - (batch, time, channels)
        # Get the logits.
        logits = self.lm_head(x)  # (linear layer) - (batch, time, vocab size)

        if targets is None:  # Training mode.
            loss = None
        else:  # Generation mode.
            # Get the last token.
            batch, time, channels = logits.shape
            # Flatten the logits.
            logits = logits.view(batch * time, channels)
            # Flatten the targets.
            targets = targets.view(batch * time)
            # Get the loss.
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    # Generate text.
    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop index to the last block_size tokens.
            index_cond = index[:, block_size:]
            # Get the logits.
            logits, _ = self.forward(index_cond)
            # Get the last token.
            logits = logits[:, -1, :]
            # Get the probabilities.
            probabilities = F.softmax(logits, dim=-1)
            # Get the index of the next token.
            index_next = torch.multinomial(probabilities, num_samples=1)
            # Append the index of the next token to the index.
            index = torch.cat((index, index_next), dim=1)
        return index


# Create the model.
m = GPTLanguageModel(vocab_size)
print("Loading model parameters...")
if os.path.isfile('model-01.pkl'):
    with open('model-01.pkl', 'rb') as f:
        m = pickle.load(f)
print("Model parameters loaded.")
# Move the model to the device.
model = m.to(device)

# Create the optimizer.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model.
for i in range(max_iterations):
    # Print the training loss.
    if i % eval_iterations == 0:
        losses = estimate_loss()
        # We want to see convergence: Val loss should be lower than train loss.
        print(f"step: {i}, train loss: {losses['train']:.3f}, val losses: {losses['val']:.3f}")

    # Get the batch.
    xb, yb = get_batch("train")
    # Forward pass.
    logits, loss = model.forward(xb, yb)
    # Backward pass.
    optimizer.zero_grad(set_to_none=True)
    # Backpropagate the loss.
    loss.backward()
    # Update the weights.
    optimizer.step()

# Print the loss.
print(loss.item())

with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved.")

