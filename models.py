import torch
import torch.nn as nn
import torchvision.models as models

class SimCLR(nn.Module):
    def __init__(self, base_model='resnet18', out_dim=128):
        super(SimCLR, self).__init__()
        # Load backbone resnet
        if base_model == 'resnet18':
            self.backbone = models.resnet18(weights=None)
        else:
            self.backbone = models.resnet34(weights=None)
            
        self.feature_dim = self.backbone.fc.in_features
        # Remove original classification head
        self.backbone.fc = nn.Identity()

        # Non-linear Projection Head (MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, out_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        return h, z


class LightweightMAE(nn.Module):
    """
    A highly illustrative patch-based autoencoder mimicking ViT-MAE logic.
    Splits image into non-overlapping patches, masks a subset, and reconstructs.
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128, mask_ratio=0.75):
        super(LightweightMAE, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        self.patch_dim = in_channels * patch_size * patch_size

        # Encoder: Linear Projection of patches
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.encoder_transformer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=256, batch_first=True)
        
        # Decoder
        self.decoder_linear = nn.Linear(embed_dim, self.patch_dim)

    def patchify(self, imgs):
        """Splits structural images into flattened vector patches."""
        p = self.patch_size
        b, c, h, w = imgs.shape
        x = imgs.reshape(b, c, h//p, p, w//p, p)
        x = torch.einsum('bchpwq->bhwcpq', x)
        x = x.reshape(b, self.num_patches, self.patch_dim)
        return x

    def unpatchify(self, x):
        """Reassembles flat vector patches back into an image tensor."""
        p = self.patch_size
        h = w = self.img_size // p
        b = x.shape[0]
        x = x.reshape(b, h, w, 3, p, p)
        x = torch.einsum('bhwcpq->bchpwq', x)
        imgs = x.reshape(b, 3, self.img_size, self.img_size)
        return imgs

    def forward(self, imgs):
        b, c, h, w = imgs.shape
        patches = self.patchify(imgs) # (B, Num_Patches, Patch_Dim)
        tokens = self.patch_embed(patches) # (B, Num_Patches, Embed_Dim)

        # Apply random masking
        num_masked = int(self.num_patches * self.mask_ratio)
        rand_indices = torch.rand(b, self.num_patches, device=imgs.device).argsort(dim=1)
        keep_indices = rand_indices[:, :self.num_patches - num_masked]
        
        # Gather unmasked structural tokens
        batch_idx = torch.arange(b, device=imgs.device).unsqueeze(1)
        latent_representations = tokens[batch_idx, keep_indices]
        
        # Pass encoded patches through Transformer blocks
        encoded = self.encoder_transformer(latent_representations)
        
        # Reconstruct missing spatial configurations using dummy tokens filled dynamically
        full_space = torch.zeros(b, self.num_patches, encoded.shape[-1], device=imgs.device)
        full_space[batch_idx, keep_indices] = encoded
        
        # Decode patches back to original dimension dimensions
        decoded_patches = self.decoder_linear(full_space)
        reconstructed_imgs = self.unpatchify(decoded_patches)
        
        # Compute dynamic features for downstream latent evaluation
        latent_features = tokens.mean(dim=1) 
        
        return reconstructed_imgs, patches, latent_features