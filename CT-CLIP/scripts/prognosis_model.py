import torch
import torch.nn as nn
from utils import MTLR

from peft import get_peft_model

class prognosis_model(nn.Module):
    def __init__(self, clip, num_time_bins, device):
        super(prognosis_model, self).__init__()
        self.clip = clip
        self.fc = nn.Linear(512, 1024)
        self.fc_2 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.mtlr = MTLR(in_features=512, num_time_bins=num_time_bins)
        self.device = device

    def forward(self, text, image):
        with torch.no_grad():
            emb = self.clip(text, image, self.device, prognosis = True)
        emb = self.fc(emb)
        # add a relu layer here
        emb = self.relu(emb)
        emb = self.fc_2(emb)
        pred = self.mtlr(emb)
        return pred

class mtlr_model(nn.Module):
    def __init__(self, clip, num_time_bins, device):
        super(mtlr_model, self).__init__()
        self.mtlr = MTLR(in_features=512, num_time_bins=num_time_bins)
        self.device = device

    def forward(self, text, image):
        with torch.no_grad():
            emb = self.clip(text, image, self.device, prognosis = True)
        emb = self.fc(emb)
        # add a relu layer here
        emb = self.relu(emb)
        emb = self.fc_2(emb)
        pred = self.mtlr(emb)
        return pred

class emb_gen(nn.Module):
    def __init__(self, clip, device):
        super(emb_gen, self).__init__()
        self.clip = clip
        self.device = device
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, text, image):
        with torch.no_grad():
            img, text = self.clip(text, image, self.device, prognosis = True)
        
        img = self.avgpool(img.permute(0, 3, 1, 2)).squeeze(-1).squeeze(-1)
        text = text.mean(dim=1)

        return img, text

class model_ctpt(nn.Module):
    def __init__(self, clip, device, num_time_bins):
        super().__init__()
        self.clip = clip
        self.device = device
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.img_embd = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )

        self.text_embd = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )

        self.fuse = nn.Sequential(
            nn.Conv1d(
                in_channels=512 * 2,
                out_channels=512,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1,
            ),
        )

        self.mtlr = MTLR(in_features=512, num_time_bins=num_time_bins)

    def forward(self, text, image, pt_image):
        with torch.no_grad():
            img, text = self.clip(text, image, pt_image, self.device, prognosis = True)
        
        img = self.avgpool(img.permute(0, 3, 1, 2)).squeeze(-1).squeeze(-1)
        text = text.mean(dim=1)

        img = self.img_embd(img)
        text = self.text_embd(text)

        fuse = torch.cat([img, text], dim=1) 
        fuse = self.fuse(fuse.unsqueeze(2))
        pred = self.mtlr(fuse.squeeze(2))

        return pred

class embd_model_new(nn.Module):
    def __init__(self, num_time_bins):
        super().__init__()

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.img_embd = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.GELU(),
        #     nn.Linear(512, 512),
        #     nn.LayerNorm(512),
        # )

        self.text_embd = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )

        # self.fuse = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=512 * 2,
        #         out_channels=512,
        #         kernel_size=3,
        #         padding=1,
        #     ),
        #     nn.GELU(),
        #     nn.Conv1d(
        #         in_channels=512,
        #         out_channels=512,
        #         kernel_size=3,
        #         padding=1,
        #     ),
        # )

        self.mtlr = MTLR(in_features=512, num_time_bins=num_time_bins)


    def forward(self, img, text):
        # self.clip.eval()
        # with torch.no_grad():
        #     img, text = self.clip(text, image, self.device, prognosis = True)

        # img = self.avgpool(img.permute(0, 3, 1, 2)).squeeze(-1).squeeze(-1)
        # text = text.mean(dim=1)

        # img = self.img_embd(img)
        text = self.text_embd(text)

        # fuse = torch.cat([img, text], dim=1) 
        # fuse = self.fuse(fuse.unsqueeze(2))
        # pred = self.mtlr(fuse.squeeze(2))

        pred = self.mtlr(text)
        
        return pred

class embd_model(nn.Module):
    def __init__(self, num_time_bins):
        super().__init__()
        # super(embd_model, self).__init__()
        self.fc = nn.Linear(512, 1024)
        self.fc_2 = nn.Linear(1024, 512)
        self.fc_3 = nn.Linear(512, 256)
        # self.fc_4 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.mtlr = MTLR(in_features=256, num_time_bins=num_time_bins)
        # self.mtlr = MTLR(in_features=128, num_time_bins=num_time_bins)
        self.batch_norm = nn.BatchNorm1d(1024)
        # self.batch_norm = nn.BatchNorm1d(512)
        # self.batch_norm = nn.BatchNorm1d(256)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # def forward(self, image, text):
    #     # emb = image + text
    #     emb = torch.cat((image, text), dim=1)
    #     # emb = text
    #     # emb = self.fc(emb)
    #     # emb = self.batch_norm(emb)
    #     # emb = self.relu(emb)
    #     # emb = self.dropout(emb)
    #     emb = self.fc_2(emb)
    #     emb = self.batch_norm(emb)
    #     emb = self.relu(emb)
    #     emb = self.dropout(emb)
    #     emb = self.fc_3(emb)
    #     emb = self.relu(emb)
    #     emb = self.dropout(emb)
    #     pred = self.mtlr(emb)
    #     return pred

    def forward(self, image, text):
        # emb = image + text
        # emb = torch.cat((text, text), dim=1)
        emb = text
        emb = self.fc(emb)
        emb = self.batch_norm(emb)
        emb = self.relu(emb)
        emb = self.dropout(emb)
        emb = self.fc_2(emb)
        emb = self.relu(emb)
        emb = self.dropout(emb)
        emb = self.fc_3(emb)
        # emb = self.batch_norm(emb)
        emb = self.relu(emb)
        emb = self.dropout(emb)
        # emb = self.fc_4(emb)
        # emb = self.relu(emb)
        # emb = self.dropout(emb)
        pred = self.mtlr(emb)
        return pred

class lora_model(nn.Module):
    def __init__(self, clip, device, peft_config, num_time_bins):
        super(lora_model, self).__init__()
        self.device = device
        self.clip = get_peft_model(clip, peft_config)
        # self.clip = clip
        self.fc = nn.Linear(512, 1024)
        self.fc_2 = nn.Linear(1024, 512)
        self.fc_3 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.mtlr = MTLR(in_features=256, num_time_bins=num_time_bins)
        self.batch_norm = nn.BatchNorm1d(1024)

        self.initialize_weights()

    def initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Check if the module name belongs to the last linear layers
                if name in ['fc', 'fc_2', 'fc_3']:  # Add the specific layer names here
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, text, image):
        self.clip.eval()
        with torch.no_grad():
            img, txt = self.clip(text, image, self.device, embed = True)
        
        # emb = torch.cat((img, txt), dim=1)
        emb = img + txt
        emb = self.fc(emb)
        emb = self.batch_norm(emb)
        emb = self.relu(emb)
        emb = self.dropout(emb)
        emb = self.fc_2(emb)
        # emb = self.batch_norm(emb)
        emb = self.relu(emb)
        emb = self.dropout(emb)
        emb = self.fc_3(emb)
        emb = self.relu(emb)
        emb = self.dropout(emb)
        pred = self.mtlr(emb)
        return pred