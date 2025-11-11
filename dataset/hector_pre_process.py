import os
import nibabel as nib
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing import Pool
from tqdm import tqdm

def read_nii_ct_files(directory):
    nii_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if 'ct' in file and file.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, file))  
    return nii_files

def read_nii_data(file_path):
    try:
        nii_img = nib.load(file_path)
        nii_data = nii_img.get_fdata()
        return nii_data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def process_file(file_path):
    img_data = read_nii_data(file_path)
    print(img_data.shape, file_path, img_data.min(), img_data.max())
    if img_data is None:
        print(f"Read {file_path} unsuccessful. Passing")
        return

    file_name = os.path.basename(file_path)

    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)
    img_data = ((img_data / 1000)).astype(np.float32)

    img_data = img_data.transpose(2, 0, 1)
    tensor = torch.tensor(img_data)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    resized_array = tensor[0][0]
    # print("reszixed", resized_array.shape, resized_array.min(), resized_array.max())

    save_folder = "/share/sda/mohammadqazi/project/hector/pre_processed/" #save folder for preprocessed
    folder_path_new = os.path.join(save_folder, "valid_" + file_name.split("_")[1], "valid_" + file_name.split("_")[1] + file_name.split("_")[2]) #folder name for train or validation
    os.makedirs(folder_path_new, exist_ok=True)
    file_name = file_name.split(".")[0]+".npz"
    save_path = os.path.join(folder_path_new, file_name)
    # print(f"Saving to {save_path}")
    np.savez(save_path, resized_array)

# Example usage:
if __name__ == "__main__":
    split_to_preprocess = '/share/sda/mohammadqazi/project/hector/dataset/processed_samples_all' 
    nii_files = read_nii_ct_files(split_to_preprocess)

    num_workers = 18  # Number of worker processes

    # Process files using multiprocessing with tqdm progress bar
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_file, nii_files), total=len(nii_files)))
