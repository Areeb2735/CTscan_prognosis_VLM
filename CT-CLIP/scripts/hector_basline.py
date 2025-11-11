# from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
from monai.data import (
    DataLoader,
    Dataset,
    decollate_batch,
)
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    MapTransform,
    ScaleIntensityd,
    SpatialPadd,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    ConcatItemsd,
    AdjustContrastd, 
    Rand3DElasticd,
    HistogramNormalized,
    NormalizeIntensityd,
    ScaleIntensityRangeD,

)

exp_name = "baseline_12_heads_fold_1_new"
fold = 1
print("--------------", exp_name, "--------------")

class HectorMONAIDataset_segmentation:
    def __init__(self, data_folder, csv_file, seg_folder):
        self.data_folder = data_folder
        self.seg_folder = seg_folder
        self.dataframe = pd.read_csv(csv_file)
        self.samples = self.prepare_samples()

    def prepare_samples(self):
        samples = []
        # Iterate over each row in the CSV.
        for idx, row in tqdm(self.dataframe.iterrows()):
            patient_id = row['PatientID']
            ct_filename = patient_id + "_ct_roi.nii.gz"
            ct_filepath = os.path.join(self.data_folder, ct_filename)
            seg_filename = patient_id + "_mask_roi.nii.gz"
            seg_filepath = os.path.join(self.seg_folder, seg_filename)
            fold = row['fold']
            sample = {
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

def get_loader_segmentation(ct_path, seg_folder, csv_file, fold): 

    train_transforms = Compose(
        [
            LoadImaged(keys=["ct", "seg"], ensure_channel_first = True),
            Orientationd(keys=["ct", "seg"], axcodes="PLS"),
            ScaleIntensityRangeD(
                keys=["ct"],
                a_min=-1024,  # Clamp minimum value
                a_max=1024,  # Clamp maximum value
                b_min=-1,    # Normalize to range [-1, 1]
                b_max=1,
                clip=True  
            ),
            RandCropByPosNegLabeld(
                keys=["ct", "seg"],
                label_key="seg",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="ct",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["ct", "seg"],
                spatial_axis=[0],
                prob=0.20,
            ),
            RandFlipd(
                keys=["ct", "seg"],
                spatial_axis=[1],
                prob=0.20,
            ),
            RandFlipd(
                keys=["ct", "seg"],
                spatial_axis=[2],
                prob=0.20,
            ),
            RandRotate90d(
                keys=["ct", "seg"],
                prob=0.20,
                max_k=3,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["ct", "seg"], ensure_channel_first = True),
            Orientationd(keys=["ct", "seg"], axcodes="PLS"),
            ScaleIntensityRangeD(
                keys=["ct"],
                a_min=-1024,  # Clamp minimum value
                a_max=1024,  # Clamp maximum value
                b_min=-1,    # Normalize to range [-1, 1]
                b_max=1,
                clip=True  
            ),
        ]
    )

    hector_data = HectorMONAIDataset_segmentation(data_folder=ct_path, csv_file=csv_file, seg_folder=seg_folder)

    train_ds, val_ds = hector_data.train_val_split(fold=fold, train_transforms=train_transforms, val_transforms=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, val_loader

train_loader, test_loader = get_loader_segmentation(ct_path = '/share/sda/mohammadqazi/project/hector/dataset/processed_samples_all',
        csv_file ="/share/sda/mohammadqazi/project/CTscan_prognosis_VLM-main/docs/TNM_hector_prompts.csv",
        seg_folder = '/share/sda/mohammadqazi/project/hector/dataset/processed_samples_all',
        fold = fold)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir  = '/share/sda/mohammadqazi/project/CTscan_prognosis_VLM-main/CT-CLIP/scripts/'
from monai.networks.nets import UNETR
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference

model = UNETR(
    in_channels=1,
    out_channels=3,
    img_size=(96, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    proj_type="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)



loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["ct"].cuda(), batch["seg"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["ct"].cuda(), batch["seg"].cuda())
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(  # noqa: B038
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(test_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(root_dir, f"{exp_name}.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


max_iterations = 25000
eval_num = 5000
post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
model.load_state_dict(torch.load(os.path.join(root_dir, f"{exp_name}.pth")))