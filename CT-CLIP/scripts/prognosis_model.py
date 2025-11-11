import torch
import torch.nn as nn
from utils import MTLR

from peft import get_peft_model

class emb_gen(nn.Module):
    def __init__(self, clip, device):
        super(emb_gen, self).__init__()
        self.clip = clip
        self.device = device
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))


    def forward(self, text, image):
        with torch.no_grad():
            img, text = self.clip(text, image, self.device, prognosis = True)

        # img = self.avgpool(img.permute(0, 3, 1, 2)).squeeze(-1).squeeze(-1)
        img = self.avgpool(img.permute(0, 4, 1, 2, 3)).squeeze(-1).squeeze(-1).squeeze(-1)
        text = text.mean(dim=1)

        return img, text

class model_lora(nn.Module):
    def __init__(self, clip, device, peft_config, num_time_bins):
        super(model_lora, self).__init__()
        self.clip = get_peft_model(clip, peft_config)
        self.device = device
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.img_embd = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.GELU(),
        #     nn.Linear(512, 512),
        #     nn.LayerNorm(512),
        # )

        # self.text_embd = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.GELU(),
        #     nn.Linear(512, 512),
        #     nn.LayerNorm(512),
        # )

        self.fuse = nn.Sequential(
            nn.Conv1d(
                in_channels=512+768,
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

    def forward(self, text, image):
        
        img, text = self.clip(text, image, self.device, prognosis = True)
        
        img = self.avgpool(img.permute(0, 3, 1, 2)).squeeze(-1).squeeze(-1)
        text = text.mean(dim=1)

        # img = self.img_embd(img)
        # text = self.text_embd(text)

        fuse = torch.cat([img, text], dim=1) 
        fuse = self.fuse(fuse.unsqueeze(2))
        pred = self.mtlr(fuse.squeeze(2))

        return pred

class model_lora_again(nn.Module):
    def __init__(self, clip, device, peft_config, num_time_bins):
        super().__init__()

        self.clip = get_peft_model(clip, peft_config)
        self.clip = clip
        self.device = device
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

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
            nn.Linear(512 + 512, 512),
            # nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        # self.mtlr = MTLR(in_features=128, num_time_bins=num_time_bins)
        self.final_layer = nn.Linear(128, num_time_bins)

    def forward(self, img, text):

        img, text = self.clip(text, img, self.device, prognosis = True)
        
        img = self.avgpool(img.permute(0, 4, 1, 2, 3)).squeeze(-1).squeeze(-1).squeeze(-1)
        text = text.mean(dim=1)

        img = self.img_embd(img)
        text = self.text_embd(text)

        fuse = torch.cat([img, text], dim=1) 

        fuse = self.fuse(fuse)
        # fuse = self.fuse(text)

        # pred = self.mtlr(fuse)  
        pred = self.final_layer(fuse)
        
        return pred

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
            nn.Linear(512 + 512, 512),
            # nn.Conv1d(
            #     in_channels=(512 + 512),
            #     out_channels=512,
            #     kernel_size=3,
            #     padding=1,
            # ),
            nn.GELU(),
            nn.Linear(512, 128),
            # nn.Conv1d(
            #     in_channels=512,
            #     out_channels=128,
            #     kernel_size=3,
            #     padding=1,
            # ),
        )

        self.mtlr = MTLR(in_features=128, num_time_bins=num_time_bins)


    def forward(self, img, text):

        img = self.img_embd(img)
        text = self.text_embd(text)

        fuse = torch.cat([img, text], dim=1) 

        # fuse = self.fuse(fuse.unsqueeze(2))
        fuse = self.fuse(fuse)
        # fuse = self.fuse(text)

        # pred = self.mtlr(fuse.squeeze(2))
        pred = self.mtlr(fuse)

        # pred = self.mtlr(text)
        # pred = self.mtlr(img)
        
        return pred

class embd_model_linear_with_adapter(nn.Module):
    def __init__(self, num_time_bins):
        super().__init__()

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
            nn.Linear(512 + 512, 512),
            # nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        # self.mtlr = MTLR(in_features=128, num_time_bins=num_time_bins)
        self.final_layer = nn.Linear(128, num_time_bins)

    def forward(self, img, text):

        img = self.img_embd(img)
        text = self.text_embd(text)

        fuse = torch.cat([img, text], dim=1) 

        fuse = self.fuse(fuse)
        # fuse = self.fuse(text)

        # pred = self.mtlr(fuse)
        pred = self.final_layer(fuse)
        
        return pred

class embd_model_linear_without_adapter(nn.Module):
    def __init__(self, num_time_bins):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.GELU(),
            nn.Linear(512, 128),
        )

        # self.mtlr = MTLR(in_features=128, num_time_bins=num_time_bins)
        self.final_layer = nn.Linear(128, num_time_bins)

    def forward(self, img, text):

        fuse = torch.cat([img, text], dim=1) 

        fuse = self.fuse(fuse)

        # pred = self.mtlr(fuse)
        pred = self.final_layer(fuse)
        
        return pred

class embd_model_conv_without_adapter(nn.Module):
    def __init__(self, num_time_bins):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Conv1d(
                in_channels=(512 + 768),
                out_channels=512,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=512,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
        )
        
        self.mtlr = MTLR(in_features=128, num_time_bins=num_time_bins)

    def forward(self, img, text):

        fuse = torch.cat([img, text], dim=1) 

        fuse = self.fuse(fuse.unsqueeze(2))
        pred = self.mtlr(fuse.squeeze(2))
        
        return pred

class embd_model_conv_with_adapter(nn.Module):
    def __init__(self, num_time_bins):
        super().__init__()

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
                in_channels=(512 + 512),
                out_channels=512,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=512,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
        )
        
        # self.mtlr = MTLR(in_features=128, num_time_bins=num_time_bins)
        self.final_layer = nn.Linear(128, num_time_bins)

    def forward(self, img, text):

        img = self.img_embd(img)
        text = self.text_embd(text)

        fuse = torch.cat([img, text], dim=1) 

        fuse = self.fuse(fuse.unsqueeze(2))

        # pred = self.mtlr(fuse.squeeze(2))
        pred = self.final_layer(fuse.squeeze(2))

        return pred
