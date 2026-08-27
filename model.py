"""
model.py
========
Task 2 — Model Architecture

Architecture justification (3-5 sentences, also required in README/report):
-----------------------------------------------------------------------
We use a U-Net-style encoder-decoder because cloud segmentation needs both
coarse semantic context (is this region cloud-like at all?) and fine
spatial precision (exact cloud boundaries), and U-Net's skip connections
pass high-resolution detail from the encoder directly to the decoder,
which a plain encoder-decoder without skips would lose. Clouds vary
hugely in scale — from small wisps to scene-covering formations — so a
multi-resolution encoder (downsampling 4x) lets the network build
features at several spatial scales instead of a single fixed receptive
field. We build it manually (no pretrained backbone) both because the
task requires a from-scratch implementation and because Landsat-8's
4-channel (R,G,B,NIR) input doesn't match the 3-channel input that
standard ImageNet-pretrained encoders expect, so pretrained weights
would not transfer cleanly anyway. The model is intentionally kept
compact (4 encoder stages, 64->512 channels) to train reasonably fast
on 384x384 patches without a very large GPU.
"""

import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    """
    Two 3x3 convolutions, each followed by BatchNorm + ReLU.

    This is the basic repeating "block" of a U-Net. Two convs (instead
    of one) give a larger effective receptive field per block without
    needing a bigger kernel, and BatchNorm stabilizes training since
    the 4 input bands have different value distributions even after
    per-band normalization.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    """
    Manual U-Net for binary cloud segmentation.

    Input:  (B, 4, H, W)  — R, G, B, NIR bands, normalized to [0, 1]
    Output: (B, 1, H, W)  — per-pixel cloud probability in [0, 1]

    Architecture:
      Encoder (contracting path): 4 stages, channels 64 -> 128 -> 256 -> 512,
        each stage = double_conv followed by 2x2 max pooling.
      Bottleneck: double_conv at the lowest resolution (1024 channels).
      Decoder (expansive path): 4 stages that upsample and concatenate
        with the matching encoder feature map (skip connection), then
        apply double_conv to fuse them.
      Output head: 1x1 conv to collapse to a single channel, then sigmoid
        to produce a probability map.
    """

    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()

        # ---------------- Encoder ----------------
        self.enc1 = double_conv(in_channels, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------------- Bottleneck ----------------
        self.bottleneck = double_conv(512, 1024)

        # ---------------- Decoder ----------------
        # ConvTranspose2d upsamples spatially by 2x while halving channels;
        # after concatenating with the skip connection, double_conv fuses
        # the upsampled features with the corresponding encoder features.
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = double_conv(512 + 512, 512)  # 512 from upconv + 512 skip

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = double_conv(256 + 256, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = double_conv(128 + 128, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = double_conv(64 + 64, 64)

        # ---------------- Output head ----------------
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ---- Encoder (save each stage's output for skip connections) ----
        e1 = self.enc1(x)              # (B, 64,  H,    W)
        p1 = self.pool(e1)             # (B, 64,  H/2,  W/2)

        e2 = self.enc2(p1)             # (B, 128, H/2,  W/2)
        p2 = self.pool(e2)             # (B, 128, H/4,  W/4)

        e3 = self.enc3(p2)             # (B, 256, H/4,  W/4)
        p3 = self.pool(e3)             # (B, 256, H/8,  W/8)

        e4 = self.enc4(p3)             # (B, 512, H/8,  W/8)
        p4 = self.pool(e4)             # (B, 512, H/16, W/16)

        # ---- Bottleneck ----
        b = self.bottleneck(p4)        # (B, 1024, H/16, W/16)

        # ---- Decoder (upsample, concat skip connection, fuse) ----
        d4 = self.upconv4(b)                     # (B, 512, H/8, W/8)
        d4 = torch.cat([d4, e4], dim=1)          # concat skip -> (B, 1024, H/8, W/8)
        d4 = self.dec4(d4)                       # (B, 512, H/8, W/8)

        d3 = self.upconv3(d4)                    # (B, 256, H/4, W/4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)                       # (B, 256, H/4, W/4)

        d2 = self.upconv2(d3)                    # (B, 128, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)                       # (B, 128, H/2, W/2)

        d1 = self.upconv1(d2)                    # (B, 64, H, W)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)                       # (B, 64, H, W)

        # ---- Output head ----
        out = self.out_conv(d1)        # (B, 1, H, W) — raw logits
        out = self.sigmoid(out)        # (B, 1, H, W) — probability map

        return out


if __name__ == "__main__":
    # quick shape sanity check — run with: python model.py
    model = UNet(in_channels=4, out_channels=1)

    dummy_input = torch.randn(2, 4, 384, 384)  # batch of 2, 4-band, 384x384
    output = model(dummy_input)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Output range: {:.3f} to {:.3f}".format(
        output.min().item(), output.max().item()
    ))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {n_params:,}")