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
from tqdm import tqdm
import torch.nn.functional as nnf

from pycox.models import CoxPH, MTLR, DeepHitSingle
from pycox import models


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
    
class Hector_Dataset_lora(Dataset):
    def __init__(self, data_folder, csv_file, args):
        self.data_folder = data_folder
        self.dataframe = pd.read_csv(csv_file)
        self.paths=[]
        self.args = args
        self.samples = self.prepare_samples()
        self.nii_to_tensor = partial(self.nii_img_to_tensor)

    def prepare_samples(self):
        samples = []
        if self.args.method == 'mtlr':
            lbltrans = MTLR.label_transform(self.args.num_time_bins, scheme='quantiles')
        elif self.args.method == 'deephit':
            lbltrans = DeepHitSingle.label_transform(self.args.num_time_bins)
        y_bins, y_events = lbltrans.fit_transform(self.dataframe['RFS'].values, self.dataframe['Relapse'].values)
        
        for index, row in self.dataframe.iterrows():
            filename = row['PatientID'] + "_ct_roi.npz"
            filepath = os.path.join(self.data_folder, filename)
            fold = row['fold']
            samples.append((filepath, filename, row['text'], row['Relapse'], row['RFS'], fold))
        samples = [tup + (val,) for tup, val in zip(samples, y_bins)]
        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path):
        img_data = np.load(path)['arr_0']
        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0)
        return tensor

    def train_val_split(self, fold):
        train_samples = []
        val_samples = []
        for sample in self.samples:
            if sample[5] == fold:
                val_samples.append(sample)
            else:
                train_samples.append(sample)
        return Hector_Dataset_lora_subset(train_samples, self), Hector_Dataset_lora_subset(val_samples, self)

    def __getitem__(self, index):
        nii_file, filename, input_text, relapse, RFS, fold, y_bin = self.samples[index]
        video_tensor = self.nii_img_to_tensor(nii_file)
        return video_tensor, input_text, relapse, RFS, filename, fold, y_bin

class Hector_Dataset_lora_subset(Dataset):
    def __init__(self, samples, parent_dataset):
        self.samples = samples
        self.parent_dataset = parent_dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        nii_file, filename, input_text, relapse, RFS, fold, y_bin = self.samples[index]
        video_tensor = self.parent_dataset.nii_img_to_tensor(nii_file)  # Calls parent method
        return video_tensor, input_text, relapse, RFS, filename, fold, y_bin

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
    def __init__(self, emd_path, csv_file, args):
        self.emd = np.load(emd_path , allow_pickle=True).item()
        self.dataframe = pd.read_csv(csv_file)
        self.paths=[]
        self.args = args
        self.samples = self.prepare_samples()
        self.nii_to_tensor = partial(self.to_tensor)

    def prepare_samples(self):
        samples = []
        if self.args.method == 'mtlr':
            lbltrans = MTLR.label_transform(self.args.num_time_bins, scheme='quantiles')
        elif self.args.method == 'deephit':
            lbltrans = DeepHitSingle.label_transform(self.args.num_time_bins)
        y_bins, y_events = lbltrans.fit_transform(self.dataframe['RFS'].values, self.dataframe['Relapse'].values)
        
        for index, row in self.dataframe.iterrows():
            filename = row['PatientID'] + "_ct_roi.nii.gz"
            # filepath = os.path.join(self.data_folder, filename)
            image_embedding = self.emd[filename]['image_embedding']
            text_embedding = self.emd[filename]['text_embedding']
            fold = row['fold']
            samples.append((image_embedding, text_embedding, row['Relapse'], row['RFS'], filename, fold))
        samples = [tup + (val,) for tup, val in zip(samples, y_bins)]
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
        image_embedding, text_embedding, relapse, RFS, filename, fold, y_bin= self.samples[index]
        text_embedding_tensor = self.to_tensor(text_embedding)
        image_embedding_tensor = self.to_tensor(image_embedding)
        return image_embedding_tensor, text_embedding_tensor, relapse, RFS, filename, fold, y_bin


import nibabel as nib

class Hector_Dataset_segmentation_emb(Dataset):
    def __init__(self, data_folder, emd_path, csv_file):
        self.data_folder = data_folder
        self.emd = np.load(emd_path , allow_pickle=True).item()
        self.dataframe = pd.read_csv(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.nii_to_tensor = partial(self.to_tensor)

    def prepare_samples(self):
        samples = []
        
        for index, row in tqdm(self.dataframe.iterrows()):
            filename = row['PatientID'] + "_ct_roi.npz"
            filepath_ct = os.path.join(self.data_folder, filename)
            hidden_state = self.emd[filename]['hidden_state']
            fold = row['fold']
            filepath_mask = os.path.join('/share/sda/mohammadqazi/project/hector/dataset/processed_samples_all/', (row['PatientID'] + "_mask_roi.nii.gz"))
            samples.append((hidden_state, filepath_ct, filepath_mask, fold))

        return samples

    def __len__(self):
        return len(self.samples)
    
    def nii_img_to_tensor(self, path):
        img_data = np.load(path)['arr_0']
        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0)
        return tensor

    def read_data(self, path_to_nifti, return_numpy=True):
        if return_numpy:
            img_data = nib.load(str(path_to_nifti)).get_fdata()
            img_data = np.expand_dims(img_data, axis=0)  # Adds channel dimension
            img_data = np.transpose(img_data, (0, 3, 1, 2))  # Reorder to (C, D, H, W)
            tensor = torch.tensor(img_data)
            # tensor = nnf.interpolate(tensor, size=torch.randn(96, 96, 96).shape, mode='trilinear', align_corners=True)
            return tensor
        return nib.load(str(path_to_nifti))

    
    def train_val_split(self, fold):
        train_samples = []
        val_samples = []
        for sample in self.samples:
            if sample[3] == fold:
                val_samples.append(sample)
            else:
                train_samples.append(sample)
        return Hector_Dataset_segmentation_emb_subset(train_samples, self), Hector_Dataset_segmentation_emb_subset(val_samples, self)

    def to_tensor(self, emb):
        # img_data = np.load(path)['arr_0']
        tensor = torch.tensor(emb)
        # tensor = tensor.unsqueeze(0)
        return tensor

    def __getitem__(self, index):
        hidden_state, mask, fold= self.samples[index]
        hidden_state_tensor = self.to_tensor(hidden_state)
        mask_tensor = self.to_tensor(mask)
        return hidden_state_tensor, mask_tensor, fold

class Hector_Dataset_segmentation_emb_subset(Dataset):
    def __init__(self, samples, parent_dataset):
        self.samples = samples
        self.parent_dataset = parent_dataset
        self.target_size = (96,96,96)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        hidden_state, filepath_ct, filepath_mask, fold = self.samples[index]
        
        mask_tensor = self.parent_dataset.read_data(filepath_mask)
        mask_tensor = mask_tensor.float()
        mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = F.interpolate(mask_tensor, size=self.target_size, mode='nearest')
        mask_tensor = mask_tensor.squeeze(0)
        mask_tensor = mask_tensor.long()

        ct_tensor = self.parent_dataset.nii_img_to_tensor(filepath_ct)
        ct_tensor = F.interpolate(ct_tensor.unsqueeze(0), size=self.target_size, mode='nearest')
        ct_tensor = ct_tensor.squeeze(0)

        return hidden_state, ct_tensor, mask_tensor, fold

