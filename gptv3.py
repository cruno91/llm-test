
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from model_gpt import get_device
from model_gpt import get_optimizer
from model_gpt import load_model
from model_gpt import train_model
from model_gpt import write_model

device = get_device()

# Hyperparameters.
# Affects memory.
batch_size = 128  # Change for GPU. (4 test, 128 train)
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


training_data_filemap = {
    "train": "bpe_output_train.txt",
    "val": "bpe_output_val.txt"
}


model = load_model(
    "model-03.pkl",
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
    multiplier=10
)

write_model("model-03.pkl", model)
