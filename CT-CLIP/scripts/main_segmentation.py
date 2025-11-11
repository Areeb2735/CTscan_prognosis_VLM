import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm, trange
import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, AdamW
from torchinfo import summary

from utils import *
from segmentation_model_again import UNETR
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


from ct_clip import CTCLIP
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from lifelines.utils import concordance_index
from data_inference_hector import Hector_Dataset_segmentation_emb
from monai_dataset import get_loader_segmentation

from pycox.models import CoxPH, MTLR, DeepHitSingle
from pycox import models

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from monai.data import decollate_batch


seed = 42
torch.manual_seed(seed) 
generator = torch.Generator().manual_seed(seed)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device {device}')

tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

text_encoder.resize_token_embeddings(len(tokenizer))
text_encoder.to(device)

image_encoder = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 480,
    patch_size = 20,
    temporal_patch_size = 10,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8
)

image_encoder.to(device)

clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_image = 294912,
    dim_text = 768,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds = False,
    use_all_token_embeds = False,
)

clip.load("docs/CT-CLIP_v2.pt")
clip.to(device)

classs=2

loss_fc = Dice_and_FocalLoss()


# loss_fc = DiceCELoss(include_background = False, to_onehot_y=True, softmax=True)
# post_label = AsDiscrete(to_onehot=classs)
# post_pred = AsDiscrete(argmax=True, to_onehot=classs)
# dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
# dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")

def main() -> None:
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--name', type=str, default='dummy', help='Name of the experiment')

    args = parser.parse_args()

    exp_name = args.name

    # hect_dataset = Hector_Dataset_segmentation_emb(data_folder = '/share/sda/mohammadqazi/project/hector/pre_processed/',
    #             emd_path = "/share/sda/mohammadqazi/project/CTscan_prognosis_VLM-main/docs/embeddings/seg.npy",  
    #             csv_file ="/share/sda/mohammadqazi/project/CTscan_prognosis_VLM-main/docs/TNM_hector_prompts.csv")

    best_dice_val_list = []


    for fold in range(5):
        print(f"Fold {fold}")

        # train_dataset, test_dataset = hect_dataset.train_val_split(fold=fold)

        # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=16)
        # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)
        
        train_loader, test_loader = get_loader_segmentation(ct_path = '/share/sda/mohammadqazi/project/hector/dataset/processed_samples_all',
                csv_file ="/share/sda/mohammadqazi/project/CTscan_prognosis_VLM-main/docs/TNM_hector_prompts.csv",
                seg_folder = '/share/sda/mohammadqazi/project/hector/dataset/processed_samples_all',
                emd_path = "/share/sda/mohammadqazi/project/CTscan_prognosis_VLM-main/docs/embeddings/seg_monai.npy",
                fold = fold)


        model = UNETR()
        # model = nn.DataParallel(model)
        model.to(device)

        # validation(model, test_loader, device, exp_name, fold)

        # hidden_state, ct_tensor, _, _ = next(iter(train_loader))
        sample = next(iter(train_loader))
        hidden_state, ct_tensor = sample['hidden_state'], sample['ct']
        hidden_state = hidden_state.to(device)
        ct_tensor = ct_tensor.to(device)
        summary(model, input_data=[ ct_tensor, hidden_state ], depth=8, col_names=["input_size", "output_size", "num_params", 'trainable'],)

        model.train()
        num_epochs = 100
        verbose=True
       
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.00001)

        # optimizer = make_optimizer(AdamW, model, lr=3e-4, weight_decay=0.00001)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
        pbar =  trange(num_epochs, disable=not verbose)
        best_dice_val = 0
        best_epoch = 0
        best_metrics = None
        epoch_loss = 0
        for i in pbar:
            model.train()
            train_dice_all = []
            # for j, (hidden_state, ct_tensor, mask_tensor, _) in tqdm(enumerate(train_loader)):
            for j, sample in tqdm(enumerate(train_loader)):
                hidden_state, ct_tensor, mask_tensor = sample['hidden_state'], sample['ct'], sample['seg']
                hidden_state = hidden_state.to(device)
                ct_tensor = ct_tensor.to(device)
                mask_tensor = mask_tensor.to(device)
                y_pred = model(ct_tensor, hidden_state)

                # os.makedirs(os.path.dirname(f'docs/weights_4/{exp_name}/{fold}/'), exist_ok=True)
                # save_input_target_prediction(ct_tensor, mask_tensor, y_pred, f'docs/weights_4/{exp_name}/{fold}/train_output_{i}_{j}.png')
                
                loss = loss_fc(y_pred, mask_tensor)
                print("Training Dice",  dice(y_pred, mask_tensor))

                optimizer.zero_grad()
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()

                train_dice_all.append(dice(y_pred, mask_tensor))
            print(sum(train_dice_all)/len(train_dice_all), "train_dice")
            pbar.set_description(f"[epoch {i+1: 4}/{num_epochs}]")
            pbar.set_postfix_str(f"loss = {loss.item():.4f}")

            if i % 5 == 0:
                dice_val = validation(model, test_loader, device, exp_name, fold)
                
            # if i > -1:
                if dice_val > best_dice_val:
                    best_dice_val = dice_val
                    best_epoch = i+1
                    save_path = os.path.join(f"docs/weights_4/{exp_name}/{fold}", 'best_model.pth')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(model.state_dict(), save_path)
                    print(f"Model saved at epoch {best_epoch}")
            
            scheduler.step()
        
        best_dice_val = round(best_dice_val, 4)
        best_dice_val_list.append(best_dice_val)

        os.makedirs(os.path.dirname(f'docs/weights_4/{exp_name}/{fold}/result.txt'), exist_ok=True)
        with open(f'docs/weights_4/{exp_name}/{fold}/result.txt', 'w') as f:
            f.write(f"Best CI: {best_dice_val:.4f}\n")
            f.write(f"Best epoch: {best_epoch}\n")
        print(f"Best CI: {best_dice_val:.4f}")
        print(f"Best epoch: {best_epoch}")
    
    # **Compute and Print Average Best CI**
    average_best_dice_val = round(sum(best_dice_val_list) / len(best_dice_val_list), 4)
    print(f"\nAverage Best CI over {len(best_dice_val_list)} folds: {average_best_dice_val:.4f}")

    # Save overall results
    with open(f'docs/weights_4/{exp_name}/final_result.txt', 'w') as f:
        f.write(f"Average Best CI: {average_best_dice_val:.4f}\n")

def validation(model, test_loader, device, exp_name, fold):
    print("Validation of the model")
    model.eval()
    dice_all = []
    with torch.no_grad():
        # for j, (hidden_state, ct_tensor, mask_tensor, _) in tqdm(enumerate(test_loader)):
        for i, sample in tqdm(enumerate(test_loader)):
            hidden_state, ct_tensor, mask_tensor = sample['hidden_state'], sample['ct'], sample['seg']
            hidden_state = hidden_state.to(device)
            ct_tensor = ct_tensor.to(device)
            mask_tensor = mask_tensor.to(device)

            y_pred = model(ct_tensor, hidden_state)
            print("Validation Dice",  dice(y_pred, mask_tensor))
            dice_all.append(dice(y_pred, mask_tensor))
        


            # os.makedirs(os.path.dirname(f'docs/weights_4/{exp_name}/{fold}/'), exist_ok=True)
            # save_input_target_prediction(ct_tensor, mask_tensor, y_pred, f'docs/weights_4/{exp_name}/{fold}/val_output_{i}.png')

            # val_labels_list = decollate_batch(mask_tensor.as_tensor())
            # val_labels_convert = [
            #     post_label(val_pred_tensor) for val_pred_tensor in val_labels_list
            # ]

            # val_outputs_list = decollate_batch(y_pred.as_tensor())
            # val_output_convert = [
            #     post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            # ]

            # dice_metric(y_pred = val_output_convert, y = val_labels_convert)
            # dice_metric_batch(y_pred=val_output_convert, y=val_labels_convert)

            # y_pred = post_pred(y_pred)
            # mask_tensor = post_label(mask_tensor)


            # dice_metric(y_pred = y_pred, y = mask_tensor)
            # dice_metric_batch(y_pred=y_pred, y=mask_tensor)
        #     print(dice_metric(y_pred = y_pred, y = mask_tensor))

        # print(dice_metric.aggregate(), "aggregate")

            # dice_all.append()

            # dice_metric(y_pred=y_pred, y=mask_tensor)
        # mean_dice_val = dice_metric.aggregate()
        # metric_batch_val = dice_metric_batch.aggregate()

        # metric_tumor = metric_batch_val[0]
        # metric_lymph = metric_batch_val[1]

        # print(f"Mean Dice Value: {mean_dice_val.item():.4f}")
        # print(f"Mean Dice Value Tumor: {metric_tumor.item():.4f}")
        # print(f"Mean Dice Value Lymph: {metric_lymph.item():.4f}")

        # dice_metric.reset()
        # dice_metric_batch.reset()

        dice_all = torch.stack(dice_all)  # creates a tensor from the list
        dice_all_np = dice_all.cpu().numpy()
        # mean_dice = dice_all_np.mean()
        # print("Mean Dice Score:", mean_dice)


        # dice_all = np.array(dice_all.cpu().numpy())
        # mean_dice_val = np.mean(dice_all)
        # dice_all_np = dice_all.cpu().numpy()
        mean_dice = dice_all_np.mean()
        print("Mean Dice Score:", mean_dice)
        # print(f"Mean Dice Value: {mean_dice_val:.4f}")
    
    # print(f"Mean Dice Value: {mean_dice:.4f}")
    return mean_dice

if __name__ == "__main__":
    main()