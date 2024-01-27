from data_archive_handler import process_data
import lzma
from tqdm import tqdm
import os
from tokenizers.implementations import ByteLevelBPETokenizer

folder_path = "../openwebtext"
output_file_train = "bpe_output_train.txt"
output_file_val = "bpe_output_val.txt"


def process_subword_training_data(file, data_split, data_source_directory):
    with open(file, "w") as outfile:
        for filename in tqdm(data_split, total=len(data_split)):
            file_path = os.path.join(data_source_directory, filename)
            with lzma.open(file_path, "rt", encoding="utf-8") as infile:
                text = infile.read()
                outfile.write(text)


# Process and concatenate the training files
files_train, files_val = process_data(folder_path)

# Process and concatenate the training files
process_subword_training_data(output_file_train, files_train, folder_path)

# Process and concatenate the validation files
process_subword_training_data(output_file_val, files_val, folder_path)

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer on the concatenated training file
tokenizer.train(files=[output_file_train], vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save the tokenizer models - vocab and merges files
tokenizer.save_model(".", "bpe_openwebtext")
