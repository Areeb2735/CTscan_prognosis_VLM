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
from prognosis_model import *
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from ct_clip import CTCLIP
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from lifelines.utils import concordance_index
from data_inference_hector import Hector_Dataset_emb

from pycox.models import CoxPH, MTLR, DeepHitSingle
from pycox import models

seed = 42
torch.manual_seed(seed) 
generator = torch.Generator().manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _pair_rank_mat(mat: np.ndarray, idx_durations: np.ndarray, events: np.ndarray, dtype: str = 'float32') -> np.ndarray:
    n = len(idx_durations)
    for i in range(n):
        dur_i = idx_durations[i]
        ev_i = events[i]
        if ev_i == 0:
            continue
        for j in range(n):
            dur_j = idx_durations[j]
            ev_j = events[j]
            if (dur_i < dur_j) or ((dur_i == dur_j) and (ev_j == 0)):
                mat[i, j] = 1
    return mat

def pair_rank_mat(idx_durations: np.ndarray, events: np.ndarray, dtype: str = 'float32') -> np.ndarray:
    """Indicator matrix R with R_ij = 1{T_i < T_j and D_i = 1}.
    So it takes value 1 if we observe that i has an event before j and zero otherwise.
    Arguments:
        idx_durations {np.array} -- Array with durations.
        events {np.array} -- Array with event indicators.
    Keyword Arguments:
        dtype {str} -- dtype of array (default: {'float32'})
    Returns:
        np.array -- n x n matrix indicating if i has an observerd event before j.
    """
    idx_durations = idx_durations.reshape(-1)
    events = events.reshape(-1)
    n = len(idx_durations)
    mat = np.zeros((n, n), dtype=dtype)
    mat = _pair_rank_mat(mat, idx_durations, events, dtype)
    return mat

def main() -> None:
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--name', type=str, default='dummy', help='Name of the experiment')
    parser.add_argument('--method', type=str, default='mtlr', help='whether to use mtlr or deephit')
    parser.add_argument('--num_time_bins', type=int, default=12, help='NUmber of time bins')

    args = parser.parse_args()

    exp_name = args.name
    method = args.method
    num_time_bins = args.num_time_bins

    hect_dataset = Hector_Dataset_emb(emd_path = 'docs/embeddings/spatial_monai_again.npy',  
                    csv_file ="docs/TNM_hector_prompts.csv", args=args)

    best_ci_list = []
    for fold in range(5):
        print(f"Fold: {fold}")

        # train_size = int(0.8 * len(hect_dataset))  # 80% for training
        # test_size = len(hect_dataset) - train_size  # 20% for testing
        # train_dataset, test_dataset = random_split(hect_dataset, [train_size, test_size], generator=generator)

        train_dataset, test_dataset = hect_dataset.train_val_split(fold=fold)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)

        if method == 'mtlr':
            loss_fc = models.loss.NLLMTLRLoss()
        elif method == 'deephit':
            loss_fc = models.loss.DeepHitSingleLoss(alpha=0.2, sigma=0.1)

        df = pd.read_csv("docs/TNM_hector_prompts.csv")
        # time_bins = make_time_bins(df['RFS'].values, event = df['Relapse'].values)
        # num_time_bins = len(time_bins)

        # num_time_bins = 12
        # time_bins = make_time_bins(df['RFS'].values, event=df['Relapse'].values, num_bins=num_time_bins)
        # lbltrans = DeepHitSingle.label_transform(num_time_bins)

        model = embd_model_linear_with_adapter(num_time_bins)
        model.to(device)

        img_emb, text_emb, relapse, RFS, _, _, _ = next(iter(train_loader))

        img_emb = img_emb.to(device)
        text_emb = text_emb.to(device)
        relapse = relapse.to(device)
        RFS = RFS.to(device)
        summary(model, input_data=[img_emb, text_emb ], depth=8, col_names=["input_size", "output_size", "num_params"],)
        
        # trainable_params = []
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         trainable_params.append(param)

        model.train()
        num_epochs = 20
        verbose=True

        trainable_params = [p for p in model.parameters() if p.requires_grad]

        optimizer = AdamW(trainable_params, lr=3e-4, weight_decay=0.00001)
        # optimizer = make_optimizer(AdamW, model, lr=3e-4, weight_decay=0.00001)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
        pbar =  trange(num_epochs, disable=not verbose)
        best_ci = 0
        best_epoch = 0
        best_metrics = None
        for i in pbar:
            model.train()
            for j, (img_emb, text_emb, relapse, RFS, _, _, y_bins) in enumerate(train_loader):
                img_emb = img_emb.to(device)
                text_emb = text_emb.to(device)
                y_bins = y_bins.to(device)
                y_pred = model(img_emb, text_emb)
                # loss = loss_fc(y_pred, torch.tensor(y_bins).to(device), relapse.to(device))
                if method == 'mtlr':
                    loss = loss_fc(y_pred, y_bins.to(device), relapse.to(device))
                elif method == 'deephit':
                    rank_mat = pair_rank_mat(y_bins, relapse)
                    loss = loss_fc(y_pred, y_bins.to(device), relapse.to(device), torch.tensor(rank_mat).to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pbar.set_description(f"[epoch {i+1: 4}/{num_epochs}]")
            pbar.set_postfix_str(f"loss = {loss.item():.4f}")
            # if i % 1 == 0:
                # torch.save(model.state_dict(), f'/home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/model_{i}.pth')
            ci, metrics = validation(model, test_loader, device)

            if i > 1:
                if ci > best_ci:
                    best_ci = ci
                    best_epoch = i+1
                    best_metrics = metrics
                    save_path = os.path.join(f"docs/weights_3/{exp_name}/{fold}", 'best_model.pth')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(model.state_dict(), save_path)
                    print(f"Model saved at epoch {best_epoch}")
            
            scheduler.step()

        best_ci = round(best_ci, 4)
        best_ci_list.append(best_ci)

        os.makedirs(os.path.dirname(f'docs/weights_3/{exp_name}/{fold}/result.txt'), exist_ok=True)
        with open(f'docs/weights_3/{exp_name}/{fold}/result.txt', 'w') as f:
            f.write(f"Best CI: {best_ci:.4f}\n")
            f.write(f"Best epoch: {best_epoch}\n")
            f.write(f"Best Metrics: {best_metrics}\n")
        print(f"Best CI: {best_ci:.4f}")
        print(f"Best epoch: {best_epoch}")
    
    # **Compute and Print Average Best CI**
    average_best_ci = round(sum(best_ci_list) / len(best_ci_list), 4)
    print(f"\nAverage Best CI over {len(best_ci_list)} folds: {average_best_ci:.4f}")

    # Save overall results
    with open(f'docs/weights_3/{exp_name}/final_result.txt', 'w') as f:
        f.write(f"Average Best CI: {average_best_ci:.4f}\n")

def validation(model, test_loader, device):
    print("Validation of the model")
    model.eval()
    pred_risk_all = []
    relapse_all = []
    RFS_all = []
    pred_survival_all = []
    with torch.no_grad():
        for img_emb, text_emb, relapse, RFS, _, _, _ in test_loader:
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
    return ci, metrics

if __name__ == "__main__":
    main()
