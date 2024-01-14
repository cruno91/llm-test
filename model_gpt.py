import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


# Define the head.
# Scaled dot production attention.
# One head of self-attention.
class Head(nn.Module):
    def __init__(self, n_embed, head_size, block_size, dropout):
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
    def __init__(self, num_heads, n_embed, head_size, block_size, dropout):
        super().__init__()
        # Module list: Heads in parallel for each head.
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size, dropout) for _ in range(num_heads)])
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
    def __init__(self, n_embed, dropout):
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
    def __init__(self, n_embed, block_size, dropout, n_head):
        # The n_embed is the amount of neurons in the embedding layer.
        # AKA the amount of channels or heads we'd like.
        super().__init__()
        # Number of features each head will be capturing in multi-head attention.
        head_size = n_embed // n_head
        # Multi-head attention layer.
        # The sa is self-attention.
        self.sa = MultiHeadAttention(n_embed, n_head, head_size, block_size, dropout)
        # Feedforward layer.
        self.ffwd = FeedForward(n_embed, dropout)
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
    def __init__(self, vocab_size, n_embed, block_size, n_head, n_layer, dropout):
        super().__init__()
        # Embedding layer.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # Positional embedding layer.
        # Each letter index has a corresponding embedding vector.
        self.positioning_embedding_table = nn.Embedding(block_size, n_embed)
        # How many decoder layers to use simultaneously.
        # Asterisk is for repeating code in the brackets.
        # Will make 4 of the "Block"s (decoder layers) simultaneously.
        self.blocks = nn.Sequential(*[Block(n_embed, block_size, dropout, n_head=n_head) for _ in range(n_layer)])
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
    def forward(self, device, index, targets=None):
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
    def generate(self, index, max_new_tokens, block_size, device):
        for _ in range(max_new_tokens):
            # Crop index to the last block_size tokens.
            index_cond = index[:, block_size:]
            # Get the logits.
            logits, loss = self.forward(device, index_cond)
            # Get the last token.
            logits = logits[:, -1, :]
            # Get the probabilities.
            probabilities = F.softmax(logits, dim=-1)
            # Get the index of the next token.
            index_next = torch.multinomial(probabilities, num_samples=1)
            # Append the index of the next token to the index.
            index = torch.cat((index, index_next), dim=1)
        return index


def load_model(model_path, vocab_size, device, n_embed, block_size, n_head, n_layer, dropout):

    # Check if the .pkl file exists
    if os.path.exists(model_path):
        print("Loading model parameters...")
        # Load the model from the .pkl file
        with open(model_path, 'rb') as f:
            m = pickle.load(f)
        print("Model parameters loaded.")
    else:
        # Handle the case where the file doesn't exist
        print(f"Model file {model_path} not found. Loading a new one.")
        # Initialize a new model instance
        m = GPTLanguageModel(vocab_size, n_embed, block_size, n_head, n_layer, dropout)

    # Set the model to the specified device
    model = m.to(device)
    return model


def get_device():
    # Check if Metal is available.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device found.")
    else:
        device = torch.device("cpu")
        print("MPS device not found.")
    return device
