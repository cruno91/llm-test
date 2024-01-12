from training_data import process_data
from training_data import process_bpe_training_data
from tokenizers.implementations import ByteLevelBPETokenizer

folder_path = "../openwebtext"
output_file_train = "bpe_output_train.txt"
output_file_val = "bpe_output_val.txt"

# Process and concatenate the training files
files_train, files_val = process_data(folder_path)

# Process and concatenate the training files
process_bpe_training_data(output_file_train, files_train, folder_path)

# Process and concatenate the validation files
process_bpe_training_data(output_file_val, files_val, folder_path)

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
