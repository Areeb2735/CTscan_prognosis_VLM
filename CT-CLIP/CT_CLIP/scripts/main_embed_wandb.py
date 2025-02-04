import pandas as pd
import numpy as np
from tqdm import tqdm, trange

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torchinfo import summary

from utils import make_time_bins
from utils import encode_survival, mtlr_neg_log_likelihood, make_optimizer
from utils import mtlr_survival, mtlr_risk, roc_auc_at_times, brier_score_at_times
from prognosis_model import embd_model

from ct_clip import CTCLIP
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from lifelines.utils import concordance_index
from data_inference_hector import Hector_Dataset_emb

import wandb

sweep_config = {
    'method': 'grid',  # Bayesian optimization
    'metric': {
        'name': 'best_ci',
        'goal': 'maximize'
    },
    'parameters': {
        # 'lr': {
        #     'min': 1e-5,
        #     'max': 1e-3,
        #     'distribution': 'log_uniform_values'
        # },
        'lr': {
            'values': [1.00000000e-05, 1.17210230e-05, 1.37382380e-05, 1.61026203e-05, 1.88739182e-05, 2.21221629e-05, 2.59294380e-05, 3.03919538e-05, 3.56224789e-05, 4.17531894e-05, 4.89390092e-05, 5.73615251e-05, 6.72335754e-05, 7.88046282e-05, 9.23670857e-05, 1.08263673e-04, 1.26896100e-04, 1.48735211e-04, 1.74332882e-04, 2.04335972e-04, 2.39502662e-04, 2.80721620e-04, 3.29034456e-04, 3.85662042e-04, 4.52035366e-04, 5.29831691e-04, 6.21016942e-04, 7.27895384e-04, 8.53167852e-04, 1.00000000e-03]
        },
        'C1': {
            'values': [1, 10, 100]
        },
        'num_time_bins': {
            'values': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            # 'min': 10,  # Starting value
            # 'max': 20,  # Maximum value
            # 'distribution': 'int_uniform'
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="ct_rate_hyperparameter_tuning_better_concat")

seed = 42
torch.manual_seed(seed) 
generator = torch.Generator().manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

clip.load("/home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/CT-CLIP/CT-CLIP_v2.pt")
clip.to(device)


hect_dataset = Hector_Dataset_emb(emd_path = '/opt/sagemaker/new_home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/save/embeddings_new_TNM.npy',  
                csv_file ="/opt/sagemaker/new_home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/TNM_hector_prompts.csv")


train_size = int(0.8 * len(hect_dataset))  # 80% for training
test_size = len(hect_dataset) - train_size  # 20% for testing
train_dataset, test_dataset = random_split(hect_dataset, [train_size, test_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

df = pd.read_csv("/opt/sagemaker/new_home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/TNM_hector_prompts.csv")
# time_bins = make_time_bins(df['RFS'].values, event = df['Relapse'].values)
# num_time_bins = len(time_bins)

def validation(model, test_loader, device):
    print("Validation of the model")
    model.eval()
    pred_risk_all = []
    relapse_all = []
    RFS_all = []
    pred_survival_all = []
    with torch.no_grad():
        for img_emb, text_emb, relapse, RFS, _ in test_loader:
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

num_epochs = 20

def train(config=None):
    # Initialize W&B run
    with wandb.init(config=config):
        config = wandb.config

        # Update num_time_bins
        global time_bins, num_time_bins
        
        num_time_bins = config.num_time_bins
        time_bins = make_time_bins(df['RFS'].values, event=df['Relapse'].values, num_bins=num_time_bins)

        model = embd_model(num_time_bins)
        model.to(device)

        optimizer = make_optimizer(Adam, model, lr=config.lr, weight_decay=0.00001)

        model.train()

        best_ci = 0
        best_epoch = 0

        for epoch in range(num_epochs):
            for img_emb, text_emb, relapse, RFS, _ in train_loader:
                img_emb, text_emb = img_emb.to(device), text_emb.to(device)
                y = encode_survival(RFS, relapse, time_bins).to(device)
                relapse, RFS = relapse.to(device), RFS.to(device)

                y_pred = model(img_emb, text_emb)
                loss = mtlr_neg_log_likelihood(y_pred, y, model, C1=config.C1, average=True)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            ci = validation(model, test_loader, device)


            # Log metrics to W&B
            wandb.log({'loss': loss.item(), 'ci': ci})

            # Save the best CI
            if epoch + 1 > 10:
                if ci > best_ci:
                    best_ci = ci
                    best_epoch = epoch+1
        
        wandb.log({'best_ci': best_ci, 'best_epoch': best_epoch})

wandb.agent(sweep_id, train, count=1000)
