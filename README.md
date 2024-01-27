# GPT LLM Testing

## Overview

This is an example GPT library written by hand by following the
[LLM from scratch](https://www.youtube.com/watch?v=UU1WVnMk4E8) tutorial by 
[Infatoshi](https://github.com/Infatoshi).

The main differences between this repository and the tutorial are mostly
the model is written as a library of classes and functions, and not as single
scripts. The purpose being to be able to have higher re-use.

I also added sub-word tokenization using the ByteLevelBPETokenizer.

Additionally, I have added logging for training as I found that tracking the
output of training to be cumbersome without it.

See [torch_notes.py](torch_notes.py) and the comments in the code, along with
the [commit log](https://github.com/cruno91/llm-test) and in my
[GPT v1 repo](https://github.com/cruno91/test-gpt-v1) for the progress I made
learning how to write LLM models using Bigram and Transformer-based 
architectures.

Metal and CUDA are supported, but I have only tested this on Apple Silicon.
There is fallback for CPU-only training, but it is not recommended.

I use PyCharm to run scripts, but you can also run them from the command line.
The [community edition of PyCharm](https://www.jetbrains.com/pycharm/download/)
is free.

## Requirements

1. Python 3.6+
2. A GPU with at least 8GB of VRAM 
   (You can adjust hyperparameters to fit your GPU)

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. [Download the openwebtext corpus](https://skylion007.github.io/OpenWebTextCorpus/) 
   (openwebtext.tar.xz)
3. Extract outside the root of this repository in an directory called 
   `openwebtext`
4. Run the `training_data_subword.py` script for generating the sub-word
   vocabulary.
5. Use the `training_data_character.py` script for generating the character
   vocabulary

## Training

My GPU is an Apple Silicon M2 Max with 64GB shared RAM.

Adapt the hyperparameters in the `gptv2.py` and `gptv3.py` scripts to fit your
GPU and desired training time.

Both the `gptv2` and `gptv3` training take about 10 hours for the default
hyperparameters with my GPU.

There are some baseline examples of hyperparameters to use in the code
comments.

I also recommend using [asitop](https://github.com/tlkh/asitop) to monitor your
GPU usage and temperature while training on Apple Silicon devices.

1. Run the `gptv3.py` script for sub-word training
2. Run the `gptv2.py` script for character training

## Results

Training logs are saved in the `logs` directory for each model like so:

- `logs/model-02/2021-03-02_22-00-00.log` for the character model
- `logs/model-03/2021-03-02_22-00-00.log` for the sub-word model

## Testing

Ensure you update the chat scripts' hyperparameters to match those from the
training scripts if you changed them.
Once you have some trained models, you can test them by running the following:

1. Run `chatv3.py` script for testing sub-word trained model
2. Run `chatv2.py` script for testing character trained model
