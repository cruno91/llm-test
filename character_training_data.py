from training_data import process_data
from training_data import process_character_training_data

folder_path = "../openwebtext"
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = "vocab.txt"

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
