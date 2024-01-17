import argparse
import torch

from model_gpt import get_device
from model_gpt import load_model

parser = argparse.ArgumentParser("Example GPT LLM.")
parser.add_argument("--batch-size", type=int, help="train the model with a specified batch size")
args = parser.parse_args()

# Hyperparameters.
# Affects memory.
batch_size = args.batch_size if args.batch_size is not None else 128  # Change for GPU. (4 test, 128 train)
block_size = 64  # Change for GPU. (v1 8 test, 64 train) - (v2 32 test, x train)
# Does not affect memory.
# Affect memory.
n_embed = 384  # Amount of neurons in the embedding layer.
n_head = 8  # Amount of heads (in parallel). (v1 4 for mps 8 for cuda) - (v2 1 test)
n_layer = 8  # Amount of layers (equal to number of decoder blocks). (v1 4 for mps 8 for cuda) - (v2 1 test)
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

device = get_device()
model = load_model("model-02.pkl", vocab_size, device, n_embed, block_size, n_head, n_layer, dropout)

while True:
    prompt = input("Enter a prompt: ")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(model.generate(context.unsqueeze(0), block_size, device, max_new_tokens=150)[0].tolist())

    print(f'Completion:\n{generated_chars}')

