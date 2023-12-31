import torch
import torch.nn as nn
import torch.nn.functional as F

# Check if Metal is available.
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found.")
else:
    device = torch.device("cpu")
    print("MPS device not found.")

# Hyperparameters.
block_size = 8
batch_size = 4
max_iters = 1000
learning_rate = 3e-3  # 0.003
eval_iters = 250

# Open the text file.
with open('wizard_of_oz.txt', 'r', encoding='utf-8') as file:
    text = file.read()
# Get the set of unique characters in the text.
chars = sorted(set(text))
vocab_size = len(chars)

# Create a tokenizer to convert between characters and numerical indices via an encoder and a decoder.
strings_to_ints = {c: i for i, c in enumerate(chars)}
encode = lambda s: [strings_to_ints[c] for c in s]
ints_to_strings = {i: c for i, c in enumerate(chars)}
decode = lambda x: ''.join([ints_to_strings[i] for i in x])

# Convert the text to integers.
data = torch.tensor(encode(text), dtype=torch.long)

# Get the training and validation splits.
n = int(0.8*len(data))
train_data, val_data = data[:n], data[n:]


# Get a batch of data.
def get_batch(split):
    # Get the data from the training or validation split.
    data = train_data if split == "train" else val_data
    # Get a random index.
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Get the data from the random index to the random index plus the block size.
    x = torch.stack([data[i:i+block_size] for i in ix])
    # Get the data from the random index plus 1 to the random index plus the block size plus 1.
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
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
        losses = torch.zeros(eval_iters)
        # Get the losses.
        for k in range(eval_iters):
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


# Define the model.
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Embedding layer.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # Forward pass.
    def forward(self, index, targets=None):
        # Get the embeddings.
        logits = self.token_embedding_table(index)

        if targets is not None:  # Training mode.
            loss = None
        else:  # Generation mode.
            # Get the last token.
            batch, time, channels = logits.shape
            # Flatten the logits.
            logits = logits.view(batch*time, channels)
            # Flatten the targets.
            targets = targets.view(batch*time)
            # Get the loss.
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    # Generate text.
    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            # Get the logits.
            logits, _ = self.forward(index)
            # Get the last token.
            logits = logits[:, -1, :]
            # Get the probabilities.
            probabilities = F.softmax(logits, dim=-1)
            # Get the index of the next token.
            index_next = torch.multinomial(probabilities, num_samples=1)
            # Append the index of the next token to the index.
            index = torch.cat((index, index_next), dim=-1)
        return index


# Create the model.
m = GPTLanguageModel(vocab_size)
# Move the model to the device.
model = m.to(device)

# Create the optimizer.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model.
for i in range(max_iters):
    # Print the training loss.
    if i % eval_iters == 0:
        losses = estimate_loss()
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
