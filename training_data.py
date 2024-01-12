import os
import lzma
from tqdm import tqdm


def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files


def process_data(folder_path):
    files = xz_files_in_dir(folder_path)
    total_files = len(files)

    split_index = int(total_files * 0.9)  # 90% of the files for training.
    files_train = files[:split_index]
    files_val = files[split_index:]

    return files_train, files_val


def process_character_training_data(file, data_split_file, folder_path, vocab):
    with open(file, "w") as outfile:
        for filename in tqdm(data_split_file, total=len(data_split_file)):
            file_path = os.path.join(folder_path, filename)
            with lzma.open(file_path, "rt", encoding="utf-8") as infile:
                text = infile.read()
                outfile.write(text)
                characters = set(text)
                vocab.update(text)
    return vocab
