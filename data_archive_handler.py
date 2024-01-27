import os


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
