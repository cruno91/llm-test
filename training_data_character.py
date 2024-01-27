from data_archive_handler import process_data
import os
import lzma
from tqdm import tqdm

folder_path = "../openwebtext"
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = "vocab.txt"


def process_character_training_data(file, data_split, data_source_directory, vocabulary):
    with open(file, "w") as outfile:
        for filename in tqdm(data_split, total=len(data_split)):
            file_path = os.path.join(data_source_directory, filename)
            with lzma.open(file_path, "rt", encoding="utf-8") as infile:
                text = infile.read()
                outfile.write(text)
                set(text)  # Characters.
                vocabulary.update(text)
    return vocabulary


# Process and concatenate the training files
files_train, files_val = process_data(folder_path)

vocab = set()

# Process the training files
vocab = process_character_training_data(output_file_train, files_train, folder_path, vocab)

# Process the validation files
vocab = process_character_training_data(output_file_val, files_val, folder_path, vocab)

# Write the vocabulary to a file.
with open(vocab_file, "w", encoding="utf-8") as vfile:
    for character in sorted(vocab):
        vfile.write(character + "\n")
