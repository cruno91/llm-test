from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from model_gpt import get_device
from model_gpt import load_model
from model_gpt import prompt

# Hyperparameters.
# Affects memory.
batch_size = 128  # Change for GPU. (4 test, 128 train)
block_size = 64  # Change for GPU. (v1 8 test, 64 train) - (v2 32 test, x train)
# Does not affect memory.
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


device = get_device()
model = load_model("model-03.pkl", vocab_size, device, n_embed, block_size, n_head, n_layer, dropout)

prompt(model, device, encode, decode, block_size)
