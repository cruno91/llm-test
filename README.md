# GPT LLM Testing

## Overview

This is an example GPT library written by hand by following the
[LLM from scratch](https://www.youtube.com/watch?v=UU1WVnMk4E8) tutorial by 
[Infatoshi](https://github.com/Infatoshi).

See [torch_notes.py](torch_notes.py) and the comments in the code, along with
the [commit log](https://github.com/cruno91/llm-test) and in my 
[GPT v1 repo](https://github.com/cruno91/test-gpt-v1) for the progress I made
learning how to write LLM models using Bigram and Transformer-based 
architectures.

I use PyCharm to run scripts, but you can also run them from the command line.

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
4. Run the `bep_training_data.py` script for generating the sub-word
   vocabulary.
5. Use the `character_training_data.py` script for generating the character 
   vocabulary

## Training

1. Run the `gptv3.py` script for sub-word training
2. Run the `gptv2.py` script for character training

My GPU is an Apple Silicon M2 Max with 64GB VRAM.

Adapt the hyper-parameters in the `gptv2.py` and `gptv3.py` scripts to fit your
GPU and desired training time.

There are some baseline exapmles of hyper-parameters to use in the code
comments.

## Results

Training logs are saved in the `logs` directory for each model like so:

- `logs/model-02/2021-03-02_22-00-00.log` for the character model
- `logs/model-03/2021-03-02_22-00-00.log` for the sub-word model

# Testing

Ensure you update the chat scripts' hyper-parameters to match those from the
training scripts if you changed them.
Once you have some trained models, you can test them by running the following:

1. Run `chatv3.py` script for testing sub-word trained model
2. Run `chatv2.py` script for testing character trained model
