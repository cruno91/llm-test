import argparse

from model_gpt import get_device
from model_gpt import load_model
from model_gpt import get_optimizer
from model_gpt import write_model
from model_gpt import train_model

device = get_device()

parser = argparse.ArgumentParser("Example GPT LLM.")
parser.add_argument("--batch-size", type=int, help="train the model with a specified batch size")
args = parser.parse_args()

# Hyperparameters.
# Affects memory.
batch_size = args.batch_size if args.batch_size is not None else 128  # Change for GPU. (4 test, 128 train)
block_size = 64  # Change for GPU. (v1 8 test, 64 train) - (v2 32 test, x train)
# Does not affect memory.
max_iterations = 65000  # Change for GPU. (v1 1000 test, 3000 train) - (v2 200 test, x train)
learning_rate = 3e-4  # 3e-3 = 0.003 - 3e-4, 1e-3, 1e-4
eval_iterations = 5000  # Change for purpose. (v1 250 test, 500 train) - (v2 100 test, x train)
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
encode = lambda s: [strings_to_ints[c] for c in s]
ints_to_strings = {i: c for i, c in enumerate(chars)}
decode = lambda x: ''.join([ints_to_strings[i] for i in x])


training_data_filemap = {
    "train": "output_train.txt",
    "val": "output_val.txt"
}

model = load_model("model-02.pkl", vocab_size, device, n_embed, block_size, n_head, n_layer, dropout)

# Create the optimizer.
optimizer = get_optimizer(model, learning_rate)

# Train the model.
train_model(model, max_iterations, optimizer, eval_iterations, training_data_filemap, block_size, batch_size, encode, device)

# Print the loss.
print(loss.item())

write_model("model-02.pkl", model)
