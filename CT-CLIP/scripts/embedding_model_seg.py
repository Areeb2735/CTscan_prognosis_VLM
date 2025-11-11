from prognosis_model import emb_gen
from data_inference_hector import Hector_Dataset
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam

from ct_clip import CTCLIP
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from lifelines.utils import concordance_index
from data_inference_hector import Hector_Dataset
from monai_dataset import get_loader_emb_gen


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

clip.load("/share/sda/mohammadqazi/project/CTscan_prognosis_VLM-main/docs/CT-CLIP_v2.pt")
clip.to(device)

# hect_dataset = Hector_Dataset(data_folder = "/share/sda/mohammadqazi/project/hector/pre_processed/",  
#                 csv_file ="docs/TNM_hector_prompts.csv")

# loader = DataLoader(hect_dataset, batch_size=8, shuffle=False)

loader = get_loader_emb_gen(ct_path = '/share/sda/mohammadqazi/project/hector/dataset/processed_samples_all',
    csv_file ="docs/TNM_hector_prompts.csv")

import numpy as np
import os
from tqdm import tqdm

embeddings_dict = {}

# model.eval()
with torch.no_grad():
    # for video, text, relapse, RFS, file_names in tqdm(loader):  # Assuming file_names is part of your dataset
    for sample in tqdm(loader):
        video, text = sample['ct'], sample['text']
        video = video.to(device)
        text_tokens = tokenizer(
            text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=512
        ).to(device)

        # Get embeddings
        img_emb, text_emb, hidden_state = clip(text_tokens, video, device, prognosis = True)
        
        hidden_state = np.stack([t.cpu().detach().numpy() for t in hidden_state], axis=1)

        # Save embeddings in the dictionary
        for i, file_name in enumerate(sample['filename']):
            embeddings_dict[file_name] = {
                'hidden_state': hidden_state[i]
            }

# Save the embeddings dictionary as a .npy file
filename = 'seg_monai.npy'
save_path = f'docs/embeddings/{filename}'
np.save(save_path, embeddings_dict)
print(f"Embeddings saved to {save_path}")
