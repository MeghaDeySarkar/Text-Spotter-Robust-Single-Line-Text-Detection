import os
import config
import pandas as pd

from utils import save_obj

if __name__ == "__main__":

    image_path = config.image_path
    label_path = config.label_path
    char2int_path = config.char2int_path
    int2char_path = config.int2char_path
    data_file_path = config.data_file_path

    # Read the labels
    labels = pd.read_table(label_path, header=None)
    # Fill missing values with "null"
    labels.fillna("null", inplace=True)

    # Get all image IDs
    image_files = os.listdir(image_path)
    image_files.sort()
    # Create full paths for the images
    image_files = [os.path.join(image_path, i) for i in image_files]

    # Find the unique characters in the labels
    unique_chars = list({l for word in labels[0] for l in word})
    unique_chars.sort()
    # Create maps from character to integer and integer to character
    char2int = {a: i+1 for i, a in enumerate(unique_chars)}
    int2char = {i+1: a for i, a in enumerate(unique_chars)}

    # Save the maps
    save_obj(char2int, char2int_path)
    save_obj(int2char, int2char_path)

    # Create data file containing the image paths and the labels
    data_file = pd.DataFrame({"images": image_files, "labels": labels[0]})
    data_file.to_csv(data_file_path, index=False)
