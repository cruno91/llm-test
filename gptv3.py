import torch
import mmap
import random
import argparse
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from model_gpt import get_device
from model_gpt import load_model
from model_gpt import get_optimizer
from model_gpt import write_model

device = get_device()

parser = argparse.ArgumentParser("Example GPT LLM.")
parser.add_argument("--batch-size", type=int, help="train the model with a specified batch size")
args = parser.parse_args()

# Hyperparameters.
# Affects memory.
batch_size = args.batch_size if args.batch_size is not None else 128  # Change for GPU. (4 test, 128 train)
block_size = 64  # Change for GPU. (v1 8 test, 64 train) - (v2 32 test, x train)
# Does not affect memory.
max_iterations = 30000  # Change for GPU. (v1 1000 test, 3000 train) - (v2 200 test, x train)
learning_rate = 3e-4  # 3e-3 = 0.003 - 3e-4, 1e-3, 1e-4
eval_iterations = 5000  # Change for purpose. (v1 250 test, 500 train) - (v2 100 test, x train)
# Affect memory.
n_embed = 384  # Amount of neurons in the embedding layer.
n_head = 8  # Amount of heads (in parallel). (v1 4 for mps 8 for cuda) - (v2 1 test)
n_layer = 8  # Amount of layers (equal to number of decoder blocks). (v1 4 for mps 8 for cuda) - (v2 1 test)
# Does not affect memory.
dropout = 0.2  # Dropout rate. 20% of the neurons will be turned off.


# Load the trained tokenizer
tokenizer = ByteLevelBPETokenizer(
    "./bpe_openwebtext-vocab.json",
    "./bpe_openwebtext-merges.txt",
)

# Post-process with BERT's way (adding special tokens, etc.)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)
vocab_size = tokenizer.get_vocab_size()


# Create a tokenizer to convert between characters and numerical indices via an encoder and a decoder.
def encode(text):
    return tokenizer.encode(text).ids


def decode(token_ids):
    return tokenizer.decode(token_ids)


# Memory map for using small snippets of text from a single file of any size.
def get_random_chunk(split):
    filename = "bpe_output_train.txt" if split == 'train' else "bpe_output_val.txt"
    # Opened in binary mode.
    with open(filename, 'rb') as f:
        # Memory map the file. (Chunks of the file are loaded into memory as needed.)
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading.
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size * batch_size)

            # Seek to the random position and read the block of text.
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size * 10 - 1)  # Increase the multiplier as needed

            # Decode the block to a string, ignoring any invalid byte sequences.
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')

            # Train and test splits.
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data


# Get a batch of data.
def get_batch(split):
    # Get the data from the training or validation split.
    while True:
        data = get_random_chunk(split)
        if len(data) > block_size:
            break
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


model = load_model("model-03.pkl", vocab_size, device, n_embed, block_size, n_head, n_layer, dropout)

# Create the optimizer.
optimizer = get_optimizer(model, learning_rate)

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
    # Backpropagation of the loss.
    loss.backward()
    # Update the weights.
    optimizer.step()

# Print the loss.
print(loss.item())

write_model("model-03.pkl", model)

