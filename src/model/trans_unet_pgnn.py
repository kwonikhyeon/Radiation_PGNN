import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=6, embed_dim=768, patch_size=16, img_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = rearrange(x, 'b c h w -> b (h w) c')  # [B, N, C]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=8, heads=12, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(embed_dim),
                nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True),
                nn.LayerNorm(embed_dim),
                nn.Sequential(
                    nn.Linear(embed_dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, embed_dim),
                    nn.Dropout(dropout),
                )]))

    def forward(self, x):
        for norm1, attn, norm2, mlp in self.layers:
            x = x + attn(norm1(x), norm1(x), norm1(x))[0]
            x = x + mlp(norm2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TransUNet(nn.Module):
    def __init__(self, in_channels=6, img_size=256, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size, img_size)
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size)**2, embed_dim))
        self.transformer = TransformerEncoder(embed_dim=embed_dim, depth=8, heads=12)

        self.upconv1 = nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2)
        self.decoder1 = DecoderBlock(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = DecoderBlock(64, 32)
        self.upconv3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder3 = DecoderBlock(16, 16)
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        x_patch = self.patch_embed(x) + self.pos_embed
        x_trans = self.transformer(x_patch)

        # reshape back to image
        H = W = int(x_patch.shape[1] ** 0.5)
        x_feat = rearrange(x_trans, 'b (h w) c -> b c h w', h=H, w=W)  # [B, C, H, W]

        x = self.upconv1(x_feat)  # 16x16 -> 32x32
        x = self.decoder1(x)
        x = self.upconv2(x)       # 32x32 -> 64x64
        x = self.decoder2(x)
        x = self.upconv3(x)       # 64x64 -> 128x128
        x = self.decoder3(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)  # up to 256x256
        return F.softplus(self.final_conv(x))