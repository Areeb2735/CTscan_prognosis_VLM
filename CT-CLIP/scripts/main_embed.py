import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm, trange

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, AdamW
from torchinfo import summary

from utils import make_time_bins
from utils import encode_survival, mtlr_neg_log_likelihood, make_optimizer
from utils import mtlr_survival, mtlr_risk, roc_auc_at_times, brier_score_at_times
from prognosis_model import embd_model, embd_model_new
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from ct_clip import CTCLIP
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from lifelines.utils import concordance_index
from data_inference_hector import Hector_Dataset_emb

seed = 42
torch.manual_seed(seed) 
generator = torch.Generator().manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
# text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

# text_encoder.resize_token_embeddings(len(tokenizer))
# text_encoder.to(device)

# image_encoder = CTViT(
#     dim = 512,
#     codebook_size = 8192,
#     image_size = 480,
#     patch_size = 20,
#     temporal_patch_size = 10,
#     spatial_depth = 4,
#     temporal_depth = 4,
#     dim_head = 32,
#     heads = 8
# )

# image_encoder.to(device)

# clip = CTCLIP(
#     image_encoder = image_encoder,
#     text_encoder = text_encoder,
#     dim_image = 294912,
#     dim_text = 768,
#     dim_latent = 512,
#     extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
#     use_mlm=False,
#     downsample_image_embeds = False,
#     use_all_token_embeds = False,
# )

# clip.load("/home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/CT-CLIP/CT-CLIP_v2.pt")
# clip.to(device)

def main() -> None:
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--name', type=str, default='dummy', help='Name of the experiment')

    args = parser.parse_args()

    exp_name = args.name

    hect_dataset = Hector_Dataset_emb(emd_path = '/opt/sagemaker/new_home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/save/embeddings_only_spatial.npy',  
                    csv_file ="/opt/sagemaker/new_home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/TNM_hector_prompts.csv")


    for fold in range(5):
        print(f"Fold: {fold}")

        # train_size = int(0.8 * len(hect_dataset))  # 80% for training
        # test_size = len(hect_dataset) - train_size  # 20% for testing
        # train_dataset, test_dataset = random_split(hect_dataset, [train_size, test_size], generator=generator)

        train_dataset, test_dataset = hect_dataset.train_val_split(fold=fold)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        df = pd.read_csv("/opt/sagemaker/new_home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/TNM_hector_prompts.csv")
        # time_bins = make_time_bins(df['RFS'].values, event = df['Relapse'].values)
        # num_time_bins = len(time_bins)

        num_time_bins = 12
        time_bins = make_time_bins(df['RFS'].values, event=df['Relapse'].values, num_bins=num_time_bins)

        model = embd_model_new(num_time_bins)
        model.to(device)

        img_emb, text_emb, relapse, RFS, _, _ = next(iter(train_loader))
        img_emb = img_emb.to(device)
        text_emb = text_emb.to(device)
        relapse = relapse.to(device)
        RFS = RFS.to(device)
        summary(model, input_data=[img_emb, text_emb ], depth=8, col_names=["input_size", "output_size", "num_params"],)

        for name, param in model.named_parameters():
            print(name, param.requires_grad)

        model.train()
        num_epochs = 20
        verbose=True
        optimizer = make_optimizer(AdamW, model, lr=3e-4, weight_decay=0.00001)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
        pbar =  trange(num_epochs, disable=not verbose)
        best_ci = 0
        best_epoch = 0
        for i in pbar:
            model.train()
            for j, (img_emb, text_emb, relapse, RFS, _, _) in enumerate(train_loader):
                img_emb = img_emb.to(device)
                text_emb = text_emb.to(device)
                y = encode_survival(RFS, relapse, time_bins).to(device)
                relapse = relapse.to(device)
                RFS = RFS.to(device)
                y_pred = model(img_emb, text_emb)
                loss = mtlr_neg_log_likelihood(y_pred, y, model, C1=10, average=True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pbar.set_description(f"[epoch {i+1: 4}/{num_epochs}]")
            pbar.set_postfix_str(f"loss = {loss.item():.4f}")
            if i % 1 == 0:
                # torch.save(model.state_dict(), f'/home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/model_{i}.pth')
                ci = validation(model, test_loader, device)
            if i > -1:
                if ci > best_ci:
                    best_ci = ci
                    best_epoch = i+1
                    save_path = os.path.join(f"/opt/sagemaker/new_home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/save/{exp_name}/{fold}", 'weight_{}.pth'.format(str(best_epoch).zfill(3)))
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(model.state_dict(), save_path)
                    print(f"Model saved at epoch {best_epoch}")
            
            scheduler.step()

        best_ci = round(best_ci, 4)
        best_epoch = best_epoch
        os.makedirs(os.path.dirname(f'/opt/sagemaker/new_home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/save/{exp_name}/{fold}/result.txt'), exist_ok=True)
        with open(f'/opt/sagemaker/new_home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/save/{exp_name}/{fold}/result.txt', 'w') as f:
            f.write(f"Best CI: {best_ci:.4f}\n")
            f.write(f"Best epoch: {best_epoch}\n")
        print(f"Best CI: {best_ci:.4f}")
        print(f"Best epoch: {best_epoch}")

def validation(model, test_loader, device):
    print("Validation of the model")
    model.eval()
    pred_risk_all = []
    relapse_all = []
    RFS_all = []
    pred_survival_all = []
    with torch.no_grad():
        for img_emb, text_emb, relapse, RFS, _, _ in test_loader:
            img_emb = img_emb.to(device)
            text_emb = text_emb.to(device)

            y_pred = model(img_emb, text_emb)
            pred_survival = mtlr_survival(y_pred).cpu().numpy()
            pred_risk = mtlr_risk(y_pred).cpu().numpy()

            pred_risk_all.append(pred_risk.item()) 
            relapse_all.append(relapse.item())
            RFS_all.append(RFS.item())
            pred_survival_all.append(list(pred_survival[0]))
    
    for i in range(len(pred_survival_all)):
        for j in range(len(pred_survival_all[i])):
            if pred_survival_all[i][j] > 1:
                pred_survival_all[i][j] = 1

    ci = concordance_index(RFS_all, -np.array(pred_risk_all), event_observed=relapse_all)
    print(f"Concordance Index: {ci:.4f}")

    eval_times = np.quantile(np.array([RFS_all[i] for i in range(len(RFS_all)) if relapse_all[i] == 1]), [.25, .5, .75]).astype(int)
    bs = brier_score_at_times(np.array(RFS_all), np.array(pred_survival_all), np.array(relapse_all), eval_times)
    auc = roc_auc_at_times(np.array(RFS_all), np.array(pred_survival_all), np.array(relapse_all), eval_times)
    metrics = []

    metrics.append({
        "model": "mtlr",
        **{f"bs_{t}": bs[i] for i, t in enumerate(eval_times)},
        **{f"auc_{t}": auc[i] for i, t in enumerate(eval_times)}
    })

    print(pd.DataFrame(metrics).round(3))
    return ci

if __name__ == "__main__":
    main()
