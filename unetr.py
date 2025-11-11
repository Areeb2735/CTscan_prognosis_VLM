from __future__ import annotations

from collections.abc import Sequence
import torch

import torch.nn as nn
import numpy as np

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
#from LoraVit import ViT, LoRA_ViT
from LoravitCTweights import ViT, LoRA_ViT
#from vit import ViT

from monai.utils import deprecated_arg, ensure_tuple_rep


class CustomedUNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        proj_type: str = "conv",
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
        r: int = 4,
        lora_layer=None,
        proj_q: bool = False,
        use_pet_encoder = False,
        use_lora = False,
        use_ct_encoder=False
        
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size. Defaults to 16.
            hidden_size: dimension of hidden layer. Defaults to 768.
            mlp_dim: dimension of feedforward layer. Defaults to 3072.
            num_heads: number of attention heads. Defaults to 12.
            proj_type: patch embedding layer type. Defaults to "conv".
            norm_name: feature normalization type and arguments. Defaults to "instance".
            conv_block: if convolutional block is used. Defaults to True.
            res_block: if residual block is used. Defaults to True.
            dropout_rate: fraction of the input units to drop. Defaults to 0.0.
            spatial_dims: number of spatial dims. Defaults to 3.
            qkv_bias: apply the bias term for the qkv linear layer in self attention block. Defaults to False.
            save_attn: to make accessible the attention in self attention block. Defaults to False.

        .. deprecated:: 1.4
            ``pos_embed`` is deprecated in favor of ``proj_type``.

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), proj_type='conv', norm_name='instance')

        """
        super().__init__() 

        print("Nada's version of UNETR")

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        


        print(img_size )
        print(self.patch_size)
        print(self.feat_size )
        print(self.# The above code is declaring a variable named "hidden_size" in Python.
        hidden_size)
        print ('zaz w dakchi ya s lbnat')
        #print("ana daba kandir ghir pet dial mda fjdid")
        
        
        self.use_lora = use_lora
        self.use_pet_encoder = use_pet_encoder
        self.use_ct_encoder = use_ct_encoder
        
        vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
            
        )
       
        # Conditionally apply LoRA
        if self.use_lora:
            self.vit = LoRA_ViT(
               vit_model=vit,
               lora_layer=lora_layer,
               r=r,
            )
        else:
            self.vit = vit  # Use the original ViT model without LoRA

        if self.use_ct_encoder:
            self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        if self.use_pet_encoder:
            self.encoder1_pt = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]


    # def proj_feat(self, x):
    #     proj_view_shape = list(x.shape[1:])  # all dimensions of x except the first one
    #     new_view = [x.size(0)] + proj_view_shape
    #     x = x.view(new_view)
    
    # # Create a new list of axes that matches the number of dimensions in x
    #     proj_axes = [i for i in self.proj_axes if -x.ndim <= i < x.ndim]
    
    #     x = x.permute(proj_axes).contiguous()
    #     return x

    # def proj_feat(self, x):
        

    #     proj_view_shape = self.proj_view_shape.copy()
    #     proj_view_shape[0] = x.size(0)  # all dimensions of x except the first one
    #     new_view = [x.size(0)] + proj_view_shape
    #     x = x.view(new_view)
    #     x = x.permute(self.proj_axes).contiguous()
    #     return x
        
    # def proj_feat(self, x):
    #     # Initial shape checks and adjustments
    #     batch_size, channels, *dims = x.shape
    #     total_elements = x.numel()

    #     # Calculate expected number of elements based on proj_view_shape
    #     expected_elements = batch_size * self.hidden_size * np.prod(self.proj_view_shape[1:])

    #     if total_elements != expected_elements:
    #         # Handle mismatch in expected vs. actual number of elements
    #         # This could involve dynamically adjusting proj_view_shape
    #         # Or applying necessary transformations to x to fit the expected shape
    #         raise ValueError(f"Shape mismatch: expected {expected_elements} elements, got {total_elements}.")

    #     # If the shapes are compatible, proceed with reshaping
    #     new_view = [batch_size] + self.proj_view_shape
    #     x = x.view(new_view)
    #     x = x.permute(self.proj_axes).contiguous()
    #     return x
    # def proj_feat(self, x):
    #     batch_size, seq_len, feature_dim = x.shape
    # # Attempt to find a reasonable reshaping strategy based on the dimensions
    # # This is just a placeholder logic - adjust as necessary for your model
    #     spatial_dim = int(np.cbrt(seq_len))

    #     if spatial_dim ** 3 != seq_len:
    #     # If exact cubic reshaping isn't possible, consider alternatives
    #     # For example, find the best fit or adjust model inputs/outputs
    #     # This might involve changing how you process or generate your data
    #         raise ValueError("Unable to reshape tensor into a cubic volume based on seq_len and feature_dim.")

    # # Proceed with reshaping if a valid cubic dimension is found
    #     new_shape = (batch_size, spatial_dim, spatial_dim, spatial_dim, feature_dim)
    #     x = x.view(new_shape)
    #     x = x.permute(0, 4, 1, 2, 3)  # Adjust based on your expected input format
    #     return x

    # def proj_feat(self, x):
    #         new_view = [x.size(0)] + self.proj_view_shape
    #         x = x.view(new_view)
    #         x = x.permute(self.proj_axes).contiguous()
    #         return x
    def proj_feat(self, x):
        
        proj_view_shape = self.proj_view_shape.copy()
        # proj_view_shape[0] = x.size(0)
        new_view = [x.size(0)] + proj_view_shape
        # In proj_feat:
        # print(f"Before reshaping x : {x.shape}")
        
        x = x.view(new_view)
        # print(f"After reshaping x: {x.shape}")
        x = x.permute(self.proj_axes).contiguous()
        return x
    
    # def proj_feat(self, x):
    #     batch_size, seq_len, _ = x.shape
    #     feature_dim = 768  # This is the hidden size or feature dimension you mentioned

    # # Calculate the correct spatial dimensions
    # # Assuming the sequence length corresponds to D*H*W where D, H, and W are spatial dimensions
    # # Here, we find D, H, W such that D*H*W*feature_dim = total number of elements in x
    #     total_elements = seq_len * feature_dim
    # # Assuming cubic dimensions for simplicity, but you may adjust this based on actual model architecture
    #     spatial_dim = int((total_elements / feature_dim) ** (1/3))

    # # Ensure the calculation above matches the expected number of elements
    #     if spatial_dim ** 3 * feature_dim != total_elements:
    #         raise ValueError("Calculated spatial dimensions do not match the tensor size.")

    # # Reshape x to the new dimensions: batch_size x spatial_dim x spatial_dim x spatial_dim x feature_dim
    #     new_view_shape = [batch_size, spatial_dim, spatial_dim, spatial_dim, feature_dim]
    #     x = x.view(new_view_shape)
    #     x = x.permute(0, 4, 1, 2, 3).contiguous()
    #     return x
        

    # Permute to adjust the tensor for the expected input format of subsequent layers if necessary
    # Adjust this

    # def proj_feat(self, x):
    #     batch_size, seq_len, feature_dim = x.shape
    #     total_elements = seq_len * feature_dim
    # # Find the closest cubic dimension that can accommodate total_elements
    #     cubic_dim = round((total_elements ** (1/3)))
    
    # Calculate the new total number of elements needed for a perfect cube
        # new_total_elements = cubic_dim ** 3
        # if new_total_elements > total_elements:
        # # If more elements are needed, pad the tensor
        #     padding_size = new_total_elements - total_elements
        #     pad_tensor = torch.zeros(batch_size, padding_size // feature_dim, feature_dim, device=x.device)
        #     x = torch.cat([x, pad_tensor], dim=1)
        # elif new_total_elements < total_elements:
        # # If fewer elements are needed, truncate the tensor
        #     x = x[:, :new_total_elements // feature_dim, :]
    
    # Now reshape x to a cubic volume with adjusted dimensions
    #     x = x.view(batch_size, cubic_dim, cubic_dim, cubic_dim, -1)
    #     return x
    # # def proj_feat(self, x):
    # #     batch_size, seq_len, _ = x.shape
    # # # Calculate the spatial dimensions dynamically based on seq_len or other criteria
    # # # Example for illustration purposes, adjust according to your model's specifics
    # #     spatial_dim = int((seq_len ** (1/3)))
    # #     new_view_shape = [batch_size, spatial_dim, spatial_dim, spatial_dim, -1]  # Adjust -1 accordingly
    
    # #     x = x.reshape(new_view_shape)
    # #     x = x.permute(0, 4, 1, 2, 3).contiguous()  # Adjust permute pattern as needed
    # #     return x

    
    def forward(self, x_in, mode=None):
        x, hidden_states_out = self.vit(x_in)
        
                            
        # Initialize encoder outputs to None
        enc1_ct = 0
        enc1_pt = 0

    # Check if the CT encoder should be used
        if self.use_ct_encoder: 
        # Extract CT scan
            ct = x_in[:, 0, :, :, :].unsqueeze(1)
                                # Use CT encoder to process CT scan
            enc1_ct = self.encoder1(ct)

        # Conditional inclusion of PET encoder based on self.use_pet_encoder flag
        if self.use_pet_encoder :
            # Extract PET scan
            pt = x_in[:, 1, :, :, :].unsqueeze(1)
                                # Use PET encoder to process PET scan
            enc1_pt = self.encoder1_pt(pt)

        enc1 = enc1_ct + enc1_pt  # Combine CT and PET encoder outputs
        # Process x2 based on mode
        x2 = hidden_states_out[3]
        #print(f"Before slicing: {x2.shape}")
        if mode == 'ct':
            x2 = x2[:, :x2.shape[1]//2, :]  # first half for ct
        elif mode == 'pt':
            x2 = x2[:, x2.shape[1]//2:, :]  # second half for pet
        elif mode == 'mix':
            x2 = x2[:, ::2, :]  # mixture both ct and pet alternating
        #print(f"After slicing: {x2.shape}")
        else:
            x2 = x2 # No slicing needed
            
        enc2 = self.encoder2(self.proj_feat(x2))
                                
        x3 = hidden_states_out[6]
        if mode == 'ct':
            x3 = x3[:, :x3.shape[1]//2, :] # first half for ct
        elif mode == 'pt':
            x3 = x3[:, x3.shape[1]//2:, :] # second half for pet
        elif mode == 'mix':
            x3 = x3[:, ::2, :] # mixture both ct and pet alternating
        else:
            x3 = x3
                                    
        enc3 = self.encoder3(self.proj_feat(x3))
                                    
        x4 = hidden_states_out[9]
        if mode == 'ct':
            x4 = x4[:, :x4.shape[1]//2, :] # first half for ct
        elif mode == 'pt':
            x4 = x4[:, x4.shape[1]//2:, :] # second half for pet
        elif mode == 'mix':
            x4 = x4[:, ::2, :] # mixture both ct and pet alternating
        else:
            x4 = x4
                                
        enc4 = self.encoder4(self.proj_feat(x4))
                                
                                # enc4 = self.encoder4(self.proj_feat(x4))
        if mode == 'ct':
            dec4 = self.proj_feat(x[:, :x.shape[1]//2, :]) # first half for ct
        elif mode == 'pt':
            dec4 = self.proj_feat(x[:, x.shape[1]//2:, :]) # second half for pet
        elif mode == 'mix':
            dec4 = self.proj_feat(x[:, ::2, :]) # mixture both ct and pet alternating
        else:
             dec4 = self.proj_feat(x)

        #dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4) # Assuming enc4 is defined in a similar way
        dec2 = self.decoder4(dec3, enc3)  # Assuming enc3 is defined
        dec1 = self.decoder3(dec2, enc2) 
        #print(dec1.shape)
        out = self.decoder2(dec1, enc1)
        #print('shape of out' , out.shape)   
        #print(dec2.shape)
        
        return self.out(out)

        


    #def forward(self, x_in, mode='ct'):
     #   x, hidden_states_out = self.vit(x_in)
        
        # Extract CT scan
       # 
        # Initialize enc1 with CT encoder output
       # enc1 = self.encoder1(ct)

        # Conditional inclusion of PET encoder based on self.use_pet_encoder flag
        #if self.use_pet_encoder and mode != 'ct':
            # Extract PET scan
          #  pt = x_in[:, 1, :, :, :].unsqueeze(1)
            # Use PET encoder to process PET scan
           # enc1_pt = self.encoder1_pt(pt)
            # Combine CT and PET encoder outputs
            # This step depends on how you want to merge these two outputs.
            # Here, I'm simply adding them, but you might want something more sophisticated.
          #  enc1 = enc1 + enc1_pt

        
      #  x2 = hidden_states_out[3]
  
       # if mode == 'ct':
          #  x2 = x2[:, :x2.shape[1]//2, :]  # first half for ct
            
           
       # elif mode == 'pt':
         #   x2 = x2[:, x2.shape[1]//2:, :]  # second half for pet
       # elif mode == 'mix':
          #  x2 = x2[:, ::2, :]  # mixture both ct and pet alternating
        
       # enc2 = self.encoder2(self.proj_feat(x2))

        #x3 = hidden_states_out[6]
        #if mode == 'ct':
         #   x3 = x3[:, :x3.shape[1]//2, :] # first half for ct
        #elif mode == 'pt':
          #  x3 = x3[:, x3.shape[1]//2:, :] # second half for pet
        #elif mode == 'mix':
        ## enc3 = self.encoder3(self.proj_feat(x3))
        #x4 = hidden_states_out[9]
        #if mode == 'ct':
          #  x4 = x4[:, :x4.shape[1]//2, :] # first half for ct
        #elif mode == 'pt':
          #  x4 = x4[:, x4.shape[1]//2:, :] # second half for pet
       # elif mode == 'mix':
          #  x4 = x4[:, ::2, :] # mixture both ct and pet alternating
        
       # enc4 = self.encoder4(self.proj_feat(x4))
        #if mode == 'ct':
           # dec4 = self.proj_feat(x[:, :x.shape[1]//2, :]) # first half for ct
        #elif mode == 'pt':
          #  dec4 = self.proj_feat(x[:, x.shape[1]//2:, :]) # second half for pet
       # elif mode == 'mix':
         #   dec4 = self.proj_feat(x[:, ::2, :]) # mixture both ct and pet alternating

       # dec3 = self.decoder5(self.proj_feat(x[:,:x.shape[1]//2,  :]), enc4)  # Assuming enc4 is defined in a similar way
       # dec2 = self.decoder4(dec3, enc3)  # Assuming enc3 is defined
       # dec1 = self.decoder3(dec2, enc2)  # Assuming enc2 is defined
      #  out = self.decoder2(dec1, enc1)
       # return self.out(out)


    # def forward(self, x_in):
    #     x, hidden_states_out = self.vit(x_in)
    #     enc1 = self.encoder1(x_in)
    #     x2 = hidden_states_out[3]
    #     enc2 = self.encoder2(self.proj_feat(x2))
    #     x3 = hidden_states_out[6]
    #     enc3 = self.encoder3(self.proj_feat(x3))
    #     x4 = hidden_states_out[9]
    #     enc4 = self.encoder4(self.proj_feat(x4))
    #     dec4 = self.proj_feat(x)
    #     dec3 = self.decoder5(dec4, enc4)
    #     dec2 = self.decoder4(dec3, enc3)
    #     dec1 = self.decoder3(dec2, enc2)
    #     out = self.decoder2(dec1, enc1)
    #     return self.out(out)
    

    
    # def forward(self, x_in, mode='ct'):
       
    #     x, hidden_states_out = self.vit(x_in)
        
    #     #2N--> N
    #     temp_x_in = x_in[:,0,:,:,:].unsqueeze(1)  
       
        
    #     enc1 = self.encoder1(temp_x_in)

    #     x2 = hidden_states_out[3]
    #     if mode == 'ct':
    #         x2 = x2[:,:x2.shape[1]//2,:] # first half for ct
    #     elif mode == 'pt':
    #         x2 = x2[:, x2.shape[1]//2:,:] # second half for pet
    #     elif mode == 'mix':
    #         x2 = x2[:, ::2, :]  # mixture both ct and pet alternating
       
    #     enc2 = self.encoder2(self.proj_feat(x2))
    #     x3 = hidden_states_out[6]
    #     if mode == 'ct':
    #         x3 = x3[:, :x3.shape[1]//2, :] # first half for ct
    #     elif mode == 'pt':
    #         x3 = x3[:, x3.shape[1]//2:, :] # second half for pet
    #     elif mode == 'mix':
    #         x3 = x3[:, ::2, :] # mixture both ct and pet alternating
       
    #     enc3 = self.encoder3(self.proj_feat(x3))
    #     x4 = hidden_states_out[9]
    #     if mode == 'ct':
    #         x4 = x4[:, :x4.shape[1]//2, :] # first half for ct
    #     elif mode == 'pt':
    #         x4 = x4[:, x4.shape[1]//2:, :] # second half for pet
    #     elif mode == 'mix':
    #         x4 = x4[:, ::2, :] # mixture both ct and pet alternating
       
    #     enc4 = self.encoder4(self.proj_feat(x4))
    #     if mode == 'ct':
    #         dec4 = self.proj_feat(x[:, :x.shape[1]//2, :]) # first half for ct
    #     elif mode == 'pt':
    #         dec4 = self.proj_feat(x[:, x.shape[1]//2:, :]) # second half for pet
    #     elif mode == 'mix':
    #         dec4 = self.proj_feat(x[:, ::2, :]) # mixture both ct and pet alternating
       
    #     dec3 = self.decoder5(dec4, enc4)
    #     dec2 = self.decoder4(dec3, enc3)
    #     dec1 = self.decoder3(dec2, enc2)
    #     out = self.decoder2(dec1, enc1)
    #     return self.out(out)

    
    # def forward(self, x_in, mode='ct'):
    #     x, hidden_states_out = self.vit(x_in)
    
    # # Separate CT and PET tokens
    #     ct_tokens = x[:, :x.shape[1]//2, :]
    #     pet_tokens = x[:, x.shape[1]//2:, :]

    # # 2N --> N
    #     temp_x_in = x_in[:,0,:,:,:].unsqueeze(1)  
    #     enc1 = self.encoder1(temp_x_in)

    # # Add PET tokens to corresponding CT tokens and apply to encoders
    #     x2 = hidden_states_out[3]
    #     enc2 = self.encoder2(self.proj_feat(ct_tokens[3] + pet_tokens[3]))
    #     x3 = hidden_states_out[6]
    #     enc3 = self.encoder3(self.proj_feat(ct_tokens[6] + pet_tokens[6]))
    #     x4 = hidden_states_out[9]
    #     enc4 = self.encoder4(self.proj_feat(ct_tokens[9] + pet_tokens[9]))

    # # Add PET tokens to CT tokens for the final decoder layer
    #     dec4 = self.proj_feat(ct_tokens[-1] + pet_tokens[-1])

    # # Decoder blocks
    #     dec3 = self.decoder5(dec4, enc4)
    #     dec2 = self.decoder4(dec3, enc3)
    #     dec1 = self.decoder3(dec2, enc2)
    #     out = self.decoder2(dec1, enc1)
    
    #     return self.out(out)


    # def forward(self, x_in, mode='ct'):
    #     x, hidden_states_out = self.vit(x_in)
        
    #     # # Assuming x_in is of shape [batch_size, channels, D, H, W]
    #     # # And channels = 2, where the first channel is CT and the second is PET
    #     # ct = x_in[:, 0, :, :, :].unsqueeze(1)  # Extract CT scan
    #     # pt = x_in[:, 1, :, :, :].unsqueeze(1)  # Extract PET scan
        
    #     # enc1 = self.encoder1(ct) if ct is not None else torch.zeros_like(self.encoder1_pt(pt))  # Add CT encoder output or empty tokens
    #     # enc1 += self.encoder1_pt(pt)  # Add PET encoder output
       
    #     x2 = hidden_states_out[3]
    #     # Check the number of channels in the input
    #     num_channels = x_in.shape[1]
    
    #     if num_channels > 1:
    #     # Extract CT and PET scans when both are present
    #         ct = x_in[:, 0, :, :, :].unsqueeze(1)
    #         pt = x_in[:, 1, :, :, :].unsqueeze(1)
    #     else:
    #     # Handle cases with only one modality
    #     # Assuming the single modality is always CT for this example
    #         ct = x_in if mode == 'ct' else None
    #         pt = x_in if mode == 'pt' else None

    # # Proceed with model processing
    # # Adjust the following lines to handle cases where either `ct` or `pt` might be None
    #     enc1 = self.encoder1(ct) if ct is not None else torch.zeros_like(self.encoder1_pt(pt))  # Example adjustment
    #     enc1_pt = self.encoder1_pt(pt) if pt is not None else torch.zeros_like(enc1)  # Adjust as necessary
        
    #     if mode == 'ct':
    #         x2 = x2[:,:x2.shape[1]//2,:] # first half for ct
    #     elif mode == 'pt':
    #         x2 = x2[:, x2.shape[1]//2:,:] # second half for pet
    #     elif mode == 'mix':
    #         x2 = x2[:, ::2, :]  # mixture both ct and pet alternating
       
    #     enc2 = self.encoder2(self.proj_feat(x2))
    #     x3 = hidden_states_out[6]
    #     if mode == 'ct':
    #         x3 = x3[:, :x3.shape[1]//2, :] # first half for ct
    #     elif mode == 'pt':
    #         x3 = x3[:, x3.shape[1]//2:, :] # second half for pet
    #     elif mode == 'mix':
    #         x3 = x3[:, ::2, :] # mixture both ct and pet alternating
       
    #     enc3 = self.encoder3(self.proj_feat(x3))
    #     x4 = hidden_states_out[9]
    #     if mode == 'ct':
    #         x4 = x4[:, :x4.shape[1]//2, :] # first half for ct
    #     elif mode == 'pt':
    #         x4 = x4[:, x4.shape[1]//2:, :] # second half for pet
    #     elif mode == 'mix':
    #         x4 = x4[:, ::2, :] # mixture both ct and pet alternating
       
    #     enc4 = self.encoder4(self.proj_feat(x4))
    #     if mode == 'ct':
    #         dec4 = self.proj_feat(x[:, :x.shape[1]//2, :]) # first half for ct
    #     elif mode == 'pt':
    #         dec4 = self.proj_feat(x[:, x.shape[1]//2:, :]) # second half for pet
    #     elif mode == 'mix':
    #         dec4 = self.proj_feat(x[:, ::2, :]) # mixture both ct and pet alternating
       
    #     dec3 = self.decoder5(dec4, enc4)
    #     dec2 = self.decoder4(dec3, enc3)
    #     dec1 = self.decoder3(dec2, enc2)
    #     out = self.decoder2(dec1, enc1)
    #     return self.out(out)

    # def forward(self, x_in, mode='ct'):
    #         x, hidden_states_out = self.vit(x_in)
            
    #         # Adjust input extraction based on available modalities
    #         num_modalities = x_in.shape[1]
    #         if num_modalities > 1:
    #             ct = x_in[:, 0, :, :, :].unsqueeze(1)  # Extract CT scan if available
    #             pet = x_in[:, 1, :, :, :].unsqueeze(1)  # Extract PET scan
    #         else:
    #             ct = None  # CT is not available
    #             pet = x_in[:, 0, :, :, :].unsqueeze(1)  # PET is assumed to be always available
            
    #         # Process encoders based on the available data
    #         enc1_ct = self.encoder1(ct) if ct is not None else torch.zeros_like(self.encoder1_pt(pet))
    #         enc1_pet = self.encoder1_pt(pet)
    #         enc1 = enc1_ct + enc1_pet
            
    #         # Apply selection based on mode to hidden states
    #         def select_half(data, half):
    #             if data is not None:
    #                 return data[:, :data.shape[1]//2, :] if half == 'first' else data[:, data.shape[1]//2:, :]
    #             return None

    #         x2 = hidden_states_out[3]
    #         x3 = hidden_states_out[6]
    #         x4 = hidden_states_out[9]
            
    #         if mode == 'ct':
    #             x2, x3, x4 = [select_half(hs, 'first') for hs in [x2, x3, x4]]
    #         elif mode == 'pt':
    #             x2, x3, x4 = [select_half(hs, 'second') for hs in [x2, x3, x4]]
    #         elif mode == 'mix':
    #             # For mix mode, we're assuming an interleaved approach. Adjust as needed.
    #             x2 = x2[:, ::2, :]  # Alternating
    #             x3 = x3[:, ::2, :]
    #             x4 = x4[:, ::2, :]

    #         # Process further encoders and decoders
    #         enc2 = self.encoder2(self.proj_feat(x2)) if x2 is not None else None
    #         enc3 = self.encoder3(self.proj_feat(x3)) if x3 is not None else None
    #         enc4 = self.encoder4(self.proj_feat(x4)) if x4 is not None else None
            
    #         # Here, you need to ensure that the decoder blocks can handle cases where `enc2`, `enc3`, or `enc4` might be `None`.
    #         # This could involve checks within those blocks or ensuring that the input to those blocks is never `None`.
    #         # For the sake of this example, we'll proceed as if those blocks can handle `None` inputs directly or have been adjusted to do so.
            
    #         dec4 = self.proj_feat(x)  # x is the output from the ViT model, adjusted for projection
    #         dec3 = self.decoder5(dec4, enc4) if enc4 is not None else dec4
    #         dec2 = self.decoder4(dec3, enc3) if enc3 is not None else dec3
    #         dec1 = self.decoder3(dec2, enc2) if enc2 is not None else dec2
    #         out = self.decoder2(dec1, enc1)  # Assuming `enc1` is always available as it's directly derived from input
            
    #         return self.out(out)
