import os
import lzma
from tqdm import tqdm
from tokenizers.implementations import ByteLevelBPETokenizer

def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

folder_path = "../openwebtext"
output_file_train = "bpe_output_train.txt"
output_file_val = "bpe_output_val.txt"

files = xz_files_in_dir(folder_path)
total_files = len(files)

split_index = int(total_files * 0.9)  # 90% of the files for training.
files_train = files[:split_index]
files_val = files[split_index:]

# Process and concatenate the training files
with open(output_file_train, "w") as outfile:
    for filename in tqdm(files_train, total=len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)

# Process and concatenate the validation files
with open(output_file_val, "w") as outfile:
    for filename in tqdm(files_val, total=len(files_val)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)

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
