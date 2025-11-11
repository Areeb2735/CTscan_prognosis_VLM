import os
import numpy as np
import pandas as pd
from monai.data import Dataset
from tqdm import tqdm
from monai.transforms import (
    Compose,
    LoadImaged,
    MapTransform,
    SpatialPadd,
    Orientationd,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    Resized,
    ScaleIntensityRangeD,
    TransposeD,
)
from torch.utils.data import DataLoader
from pycox.models import CoxPH, MTLR, DeepHitSingle

from monai.transforms import Transform
import numpy as np
import torch

class ConvertMaskD(Transform):
    """
    A custom transform that converts mask values of 2 to 0.
    This transform works for both NumPy arrays and PyTorch tensors.
    """
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                mask = data[key]
                if isinstance(mask, np.ndarray):
                    mask[mask == 2] = 0
                elif torch.is_tensor(mask):
                    mask = torch.where(mask == 2,
                                       torch.tensor(0, dtype=mask.dtype, device=mask.device),
                                       mask)
                data[key] = mask
        return data


# Custom transform to load NPZ files.
class LoadNPZd(MapTransform):
    """
    Dictionary-based transform to load an NPZ file.
    Expects the NPZ file to have an array stored under 'arr_0'
    and then adds a channel dimension.
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            file_path = d[key]
            npz_data = np.load(file_path)['arr_0']
            # If npz_data is 3D, add a channel dimension to become (1, D, H, W)
            # if npz_data.ndim == 3:
            npz_data = np.expand_dims(npz_data, axis=0)
            # npz_data = npz_data.transpose(0, 2, 3, 1)
            d[key] = npz_data
        return d

# Custom transform to keep 'hidden_state' unchanged
class IdentityTransform(MapTransform):
    """
    A MONAI transform that simply passes through the hidden state
    without modification.
    """
    def __call__(self, data):
        return data  # No modifications

class HectorMONAIDataset_segmentation:
    def __init__(self, data_folder, emd_path, csv_file, seg_folder):
        self.data_folder = data_folder
        self.seg_folder = seg_folder
        self.emd = np.load(emd_path, allow_pickle=True).item()
        self.dataframe = pd.read_csv(csv_file)
        self.samples = self.prepare_samples()

    def prepare_samples(self):
        samples = []
        # Iterate over each row in the CSV.
        for idx, row in tqdm(self.dataframe.iterrows()):
            patient_id = row['PatientID']
            # Build filepaths based on the patient ID.
            # ct_filename = patient_id + "_ct_roi.npz"
            ct_filename = patient_id + "_ct_roi.nii.gz"
            ct_filepath = os.path.join(self.data_folder, ct_filename)
            seg_filename = patient_id + "_mask_roi.nii.gz"
            seg_filepath = os.path.join(self.seg_folder, seg_filename)
            # Retrieve the hidden state from the embedding dictionary.
            hidden_state = self.emd[ct_filename]['hidden_state']
            fold = row['fold']
            # Build a dictionary sample.
            sample = {
                "hidden_state": hidden_state,  # Will be preserved
                "ct": ct_filepath,
                "seg": seg_filepath,
                "fold": fold,
            }
            samples.append(sample)
        return samples

    def train_val_split(self, fold, train_transforms: Compose, val_transforms: Compose):
        train_samples = [s for s in self.samples if s["fold"] != fold]
        val_samples = [s for s in self.samples if s["fold"] == fold]
        train_ds = Dataset(data=train_samples, transform=train_transforms)
        val_ds = Dataset(data=val_samples, transform=val_transforms)
        return train_ds, val_ds

def get_loader_segmentation(ct_path, seg_folder, emd_path, csv_file, fold): 

    train_transforms = Compose([
        LoadImaged(keys=["ct", "seg"], ensure_channel_first=True),
        TransposeD(keys=["ct", "seg"], indices=(0, 3, 1, 2)),
        ConvertMaskD(keys=["seg"]),
        IdentityTransform(keys=["hidden_state"]),  # Preserve hidden state
        ScaleIntensityRangeD(
            keys=["ct"],
            a_min=-1024,  # Clamp minimum value
            a_max=1024,  # Clamp maximum value
            b_min=-1,    # Normalize to range [-1, 1]
            b_max=1,
            clip=True    # Ensure values stay within the specified range
        ),
        Resized(keys=["seg"], spatial_size=(192, 192, 192), mode="nearest"),
        Orientationd(keys=["ct", "seg"], axcodes="RAS"),
    ])

    val_transforms = Compose([
        # LoadNPZd(keys=["ct"]),
        LoadImaged(keys=["ct", "seg"], ensure_channel_first=True),
        TransposeD(keys=["ct", "seg"], indices=(0, 3, 1, 2)),
        ConvertMaskD(keys=["seg"]),
        IdentityTransform(keys=["hidden_state"]),  # Preserve hidden state
        ScaleIntensityRangeD(
            keys=["ct"],
            a_min=-1024,  # Clamp minimum value
            a_max=1024,  # Clamp maximum value
            b_min=-1,    # Normalize to range [-1, 1]
            b_max=1,
            clip=True    # Ensure values stay within the specified range
        ),
        Resized(keys=["seg"], spatial_size=(192, 192, 192), mode="nearest"),
        Orientationd(keys=["ct", "seg"], axcodes="RAS"),
    ])

    hector_data = HectorMONAIDataset_segmentation(data_folder=ct_path, emd_path=emd_path, csv_file=csv_file, seg_folder=seg_folder)

    train_ds, val_ds = hector_data.train_val_split(fold=fold, train_transforms=train_transforms, val_transforms=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    # sample_batch = next(iter(train_loader))

    # print(sample_batch.keys())
    # print(sample_batch["hidden_state"].shape)

    return train_loader, val_loader

class HectorMONAIDataset_prognosis:
    def __init__(self, data_folder, csv_file, args):
        self.data_folder = data_folder
        self.dataframe = pd.read_csv(csv_file)
        self.args = args
        self.samples = self.prepare_samples()

    def prepare_samples(self):
        samples = []
        if self.args.method == 'mtlr':
            lbltrans = MTLR.label_transform(self.args.num_time_bins, scheme='quantiles')
        elif self.args.method == 'deephit':
            lbltrans = DeepHitSingle.label_transform(self.args.num_time_bins)
        y_bins, y_events = lbltrans.fit_transform(self.dataframe['RFS'].values, self.dataframe['Relapse'].values)
        
        for idx, row in tqdm(self.dataframe.iterrows()):
            patient_id = row['PatientID']
            ct_filename = patient_id + "_ct_roi.nii.gz"
            ct_filepath = os.path.join(self.data_folder, ct_filename)
            fold = row['fold']
            sample = {
                "ct": ct_filepath,
                'text': row['text'],
                'relapse': row['Relapse'],
                'RFS': row['RFS'],
                'filename': ct_filename,
                "fold": fold,
            }
            samples.append(sample)
        for d, new_value in zip(samples, y_bins):
            d["y_bin"] = new_value

        return samples

    def train_val_split(self, fold, train_transforms: Compose, val_transforms: Compose):
        train_samples = [s for s in self.samples if s["fold"] != fold]
        val_samples = [s for s in self.samples if s["fold"] == fold]
        train_ds = Dataset(data=train_samples, transform=train_transforms)
        val_ds = Dataset(data=val_samples, transform=val_transforms)
        return train_ds, val_ds

def get_loader_prognosis(ct_path, csv_file, fold, args): 

    train_transforms = Compose([
        LoadImaged(keys=["ct"], ensure_channel_first=True),
        TransposeD(keys=["ct"], indices=(0, 3, 1, 2)),
        ScaleIntensityRangeD(
            keys=["ct"],
            a_min=-1024,  # Clamp minimum value
            a_max=1024,  # Clamp maximum value
            b_min=-1,    # Normalize to range [-1, 1]
            b_max=1,
            clip=True    # Ensure values stay within the specified range
        ),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["ct"], ensure_channel_first=True),
        TransposeD(keys=["ct"], indices=(0, 3, 1, 2)),
        ScaleIntensityRangeD(
            keys=["ct"],
            a_min=-1024,  # Clamp minimum value
            a_max=1024,  # Clamp maximum value
            b_min=-1,    # Normalize to range [-1, 1]
            b_max=1,
            clip=True    # Ensure values stay within the specified range
        ),
    ])

    hector_data = HectorMONAIDataset_prognosis(data_folder=ct_path, csv_file=csv_file, args = args)

    train_ds, val_ds = hector_data.train_val_split(fold=fold, train_transforms=train_transforms, val_transforms=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    # sample_batch = next(iter(train_loader))

    # print(sample_batch.keys())
    # print(sample_batch["hidden_state"].shape)

    return train_loader, val_loader

class HectorMONAIDataset_emb_gen:
    def __init__(self, data_folder, csv_file):
        self.data_folder = data_folder
        self.dataframe = pd.read_csv(csv_file)
        self.samples = self.prepare_samples()

    def prepare_samples(self):
        samples = []
        for idx, row in tqdm(self.dataframe.iterrows()):
            patient_id = row['PatientID']
            ct_filename = patient_id + "_ct_roi.nii.gz"
            ct_filepath = os.path.join(self.data_folder, ct_filename)
            sample = {
                "ct": ct_filepath,
                'text': row['text'],
                'relapse': row['Relapse'],
                'RFS': row['RFS'],
                'filename': ct_filename
            }
            samples.append(sample)
        return samples

    def get_dataset(self, transforms: Compose):
        samples = [s for s in self.samples]
        ds = Dataset(data=samples, transform=transforms)
        return ds

def get_loader_emb_gen(ct_path, csv_file): 

    transforms = Compose([
        LoadImaged(keys=["ct"], ensure_channel_first=True),
        TransposeD(keys=["ct"], indices=(0, 3, 1, 2)),
        ScaleIntensityRangeD(
            keys=["ct"],
            a_min=-1024,  # Clamp minimum value
            a_max=1024,  # Clamp maximum value
            b_min=-1,    # Normalize to range [-1, 1]
            b_max=1,
            clip=True    # Ensure values stay within the specified range
        ),
        # Orientationd(keys=["ct"], axcodes="RAS"),
        # NormalizeIntensityd(keys=["ct"], channel_wise=True),
    ])

    hector_data = HectorMONAIDataset_emb_gen(data_folder=ct_path, csv_file=csv_file)

    ds = hector_data.get_dataset(transforms=transforms)

    train_loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)

    # sample_batch = next(iter(train_loader))

    # print(sample_batch.keys())
    # print(sample_batch["ct"].shape)

    return train_loader


if __name__ == "__main__":
    # fold = 0
    ct_path =  '/share/sda/mohammadqazi/project/hector/dataset/processed_samples_all'
    # seg_folder = '/share/sda/mohammadqazi/project/hector/dataset/processed_samples_all'
    # emd_path = '/share/sda/mohammadqazi/project/CTscan_prognosis_VLM-main/docs/embeddings/seg.npy'
    csv_file = '/share/sda/mohammadqazi/project/CTscan_prognosis_VLM-main/docs/TNM_hector_prompts.csv'

    # train_loader, val_loader = get_loader_segmentation(ct_path, seg_folder, emd_path, csv_file, fold)

    get_loader_emb_gen(ct_path, csv_file)