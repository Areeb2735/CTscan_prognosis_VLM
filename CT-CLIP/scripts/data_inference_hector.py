import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import tqdm


class Hector_Dataset(Dataset):
    def __init__(self, data_folder, csv_file):
        self.data_folder = data_folder
        self.dataframe = pd.read_csv(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.nii_to_tensor = partial(self.nii_img_to_tensor)

    def prepare_samples(self):
        samples = []

        for index, row in self.dataframe.iterrows():
            filename = row['PatientID'] + "_ct_roi.npz"
            filepath = os.path.join(self.data_folder, filename)
            samples.append((filepath, filename, row['text'], row['Relapse'], row['RFS']))

        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path):
        img_data = np.load(path)['arr_0']
        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0)
        return tensor

    def __getitem__(self, index):
        nii_file, filename, input_text, relapse, RFS = self.samples[index]
        video_tensor = self.nii_to_tensor(nii_file)
        return video_tensor, input_text, relapse, RFS, filename

class Hector_Dataset_ct_pt(Dataset):
    def __init__(self, data_folder, csv_file):
        self.data_folder = data_folder
        self.dataframe = pd.read_csv(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.nii_to_tensor = partial(self.nii_img_to_tensor)

    def prepare_samples(self):
        samples = []

        for index, row in self.dataframe.iterrows():
            filename_ct = row['PatientID'] + "_ct_roi.npz"
            filepath_ct = os.path.join(self.data_folder, filename_ct)
            filename_pt = row['PatientID'] + "_pt_roi.npz"
            filepath_pt = os.path.join(self.data_folder, filename_pt)
            samples.append((filepath_ct, filepath_pt, filename_ct, row['text'], row['Relapse'], row['RFS']))

        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path):
        img_data = np.load(path)['arr_0']
        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0)
        return tensor

    def __getitem__(self, index):
        nii_file_ct, nii_file_pt, filename, input_text, relapse, RFS = self.samples[index]
        ct_tensor = self.nii_to_tensor(nii_file_ct)
        pt_tensor = self.nii_to_tensor(nii_file_pt)
        return ct_tensor, pt_tensor, input_text, relapse, RFS, filename

class Hector_Dataset_emb(Dataset):
    def __init__(self, emd_path, csv_file):
        self.emd = np.load(emd_path , allow_pickle=True).item()
        self.dataframe = pd.read_csv(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.nii_to_tensor = partial(self.to_tensor)

    def prepare_samples(self):
        samples = []

        for index, row in self.dataframe.iterrows():
            filename = row['PatientID'] + "_ct_roi.npz"
            # filepath = os.path.join(self.data_folder, filename)
            image_embedding = self.emd[filename]['image_embedding']
            text_embedding = self.emd[filename]['text_embedding']
            fold = row['fold']
            samples.append((image_embedding, text_embedding, row['Relapse'], row['RFS'], filename, fold))

        return samples

    def __len__(self):
        return len(self.samples)
    
    def train_val_split(self, fold):
        train_samples = []
        val_samples = []
        for sample in self.samples:
            if sample[5] == fold:
                val_samples.append(sample)
            else:
                train_samples.append(sample)
        return train_samples, val_samples

    def to_tensor(self, emb):
        # img_data = np.load(path)['arr_0']
        tensor = torch.tensor(emb)
        # tensor = tensor.unsqueeze(0)
        return tensor

    def __getitem__(self, index):
        image_embedding, text_embedding, relapse, RFS, filename, fold = self.samples[index]
        text_embedding_tensor = self.to_tensor(text_embedding)
        image_embedding_tensor = self.to_tensor(image_embedding)
        return image_embedding_tensor, text_embedding_tensor, relapse, RFS, filename, fold
