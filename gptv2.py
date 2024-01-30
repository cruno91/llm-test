import time

from model_gpt import get_device
from model_gpt import get_optimizer
from model_gpt import load_model
from model_gpt import train_model
from model_gpt import write_model
from model_gpt import setup_logging
from model_gpt import start_log_training_session
from model_gpt import finish_log_training_session

device = get_device()

# Hyperparameters.
# Affects memory.
batch_size = 128  # Change for GPU. (4 test, 128 train)
block_size = 64  # Change for GPU. (v1 8 test, 64 train) - (v2 32 test, x train)
# Does not affect memory.
max_iterations = 500  # Change for GPU. (v1 1000 test, 3000 train) - (v2 200 test, x train)
learning_rate = 3e-4  # 3e-3 = 0.003 - 3e-4, 1e-3, 1e-4
eval_iterations = 100  # Change for purpose. (v1 250 test, 500 train) - (v2 100 test, x train)
# Affect memory.
n_embed = 384  # Amount of neurons in the embedding layer.
n_head = 8  # Amount of heads (in parallel). (v1 4 for mps 8 for cuda) - (v2 1 test)
n_layer = 8  # Amount of layers (equal to number of decoder blocks). (v1 4 for mps 8 for cuda) - (v2 1 test)
# Does not affect memory.
dropout = 0.2  # Dropout rate. 20% of the neurons will be turned off.

# 127000

# Open the text file.
with open('vocab.txt', 'r', encoding='utf-8') as file:
    text = file.read()
    # Get the set of unique characters in the text.
    chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a tokenizer to convert between characters and numerical indices via an encoder and a decoder.
strings_to_ints = {c: i for i, c in enumerate(chars)}
ints_to_strings = {i: c for i, c in enumerate(chars)}


def encode(s):
    return [strings_to_ints[c] for c in s]


def decode(x):
    return ''.join([ints_to_strings[i] for i in x])


training_data_filemap = {
    "train": "output_train.txt",
    "val": "output_val.txt"
}

model = load_model(
    "model-02.pkl",
    vocab_size,
    device,
    n_embed,
    block_size,
    n_head,
    n_layer,
    dropout
)

# Create the optimizer.
optimizer = get_optimizer(model, learning_rate)

hyperparameters = {
    "batch_size": batch_size,
    "block_size": block_size,
    "max_iterations": max_iterations,
    "learning_rate": learning_rate,
    "eval_iterations": eval_iterations,
    "n_embed": n_embed,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout
}

model_name = "model-02"

log_file_path = setup_logging(model_name)

# Track when the training starts
start_time = time.time()

start_log_training_session(log_file_path, hyperparameters)

# Train the model.
train_model(
    model,
    max_iterations,
    optimizer,
    eval_iterations,
    training_data_filemap,
    block_size,
    batch_size,
    encode,
    device,
    log_file_path,
    multiplier=1
)

# Calculate the training duration
training_duration = time.time() - start_time

finish_log_training_session(log_file_path, training_duration)

write_model(model_name + ".pkl", model)
