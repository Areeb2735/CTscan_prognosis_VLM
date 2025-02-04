import pandas as pd
import numpy as np
from tqdm import tqdm, trange

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam

from utils import make_time_bins
from utils import encode_survival, mtlr_neg_log_likelihood, make_optimizer
from utils import mtlr_survival, mtlr_risk
from prognosis_model import prognosis_model

from ct_clip import CTCLIP
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from lifelines.utils import concordance_index
from data_inference_hector import Hector_Dataset


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

hect_dataset = Hector_Dataset(data_folder = "/home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/valid_preprocessed_hector/",  
                csv_file ="/home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/final_hector_with_text.csv")


train_size = int(0.8 * len(hect_dataset))  # 80% for training
test_size = len(hect_dataset) - train_size  # 20% for testing
train_dataset, test_dataset = random_split(hect_dataset, [train_size, test_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

df = pd.read_csv("/home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/final_hector_with_text.csv")
time_bins = make_time_bins(df['RFS'].values, event = df['Relapse'].values)
num_time_bins = len(time_bins)

model = prognosis_model(clip, num_time_bins, device = device)
model.to(device)

model.load_state_dict(torch.load('/opt/sagemaker/new_home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/model_40.pth'))

model.eval()
pred_risk_all = []
relapse_all = []
RFS_all = []
with torch.no_grad():
    for video, text, relapse, RFS, _ in test_loader:
        video = video.to('cuda')
        text_tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to('cuda')
        # y = encode_survival(RFS, relapse, time_bins).to(device)

        y_pred = model(text_tokens, video)
        pred_survival = mtlr_survival(y_pred).cpu().numpy()
        pred_risk = mtlr_risk(y_pred).cpu().numpy()

        pred_risk_all.append(pred_risk.item()) 
        relapse_all.append(relapse.item())
        RFS_all.append(RFS.item())

ci = concordance_index(RFS_all, -np.array(pred_risk_all), event_observed=relapse_all)
print(f"Concordance Index: {ci:.4f}")
