import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .blocks import ConvBlock, DeconvBlock


#####################################################################
# Generator
#####################################################################
class CGenerator(nn.Module):
    def __init__(self, img_size=32, latent_dim=128, out_channels=3, base=64,
                 num_classes=10, embedding_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.embedding_dim = embedding_dim
        self.labels_embedding = nn.Embedding(num_classes, embedding_dim)

        num_blocks = {32: 2, 64: 3, 128: 4, 256: 5}
        if img_size not in num_blocks:
            raise ValueError(f"Unsupported img_size: {img_size}")

        in_channels = base * (2 ** num_blocks[img_size])
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + embedding_dim, in_channels,
                kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )

        blocks = []
        for i in range(num_blocks[img_size]):
            blocks.append(DeconvBlock(in_channels, in_channels // 2))
            in_channels //= 2
        self.blocks = nn.Sequential(*blocks)

        self.final = nn.ConvTranspose2d(base, out_channels,
                kernel_size=4, stride=2, padding=1, bias=False)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, z, labels):
        z = z.view(-1, self.latent_dim, 1, 1)
        labels = self.labels_embedding(labels).view(-1, self.embedding_dim, 1, 1)
        x = torch.cat([z, labels], dim=1)

        x = self.initial(x)
        x = self.blocks(x)
        x = self.final(x)
        return torch.tanh(x)


#####################################################################
# Discriminator
#####################################################################
class CDiscriminator(nn.Module):
    def __init__(self, img_size=32, in_channels=3, base=64,
                 num_classes=10, embedding_channels=1):
        super().__init__()
        self.img_size = img_size
        self.embedding_channels = embedding_channels
        self.labels_embedding = nn.Embedding(num_classes, embedding_channels)

        num_blocks = {32: 2, 64: 3, 128: 4, 256: 5}
        if img_size not in num_blocks:
            raise ValueError(f"Unsupported img_size: {img_size}")

        out_channels = base
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels + embedding_channels, out_channels,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        blocks = []
        for i in range(num_blocks[img_size]):
            blocks.append(ConvBlock(out_channels, out_channels * 2))
            out_channels *= 2
        self.blocks = nn.Sequential(*blocks)

        self.final = nn.Conv2d(out_channels, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, images, labels):
        # images: (B, C, H, W), labels: (B,)
        h, w = images.size(-2), images.size(-1)
        labels = self.labels_embedding(labels)                  # (B, embedding_channels)
        labels = labels.view(-1, self.embedding_channels, 1, 1) # (B, embedding_channels, 1, 1)
        labels = labels.expand(-1, -1, h, w)                    # (B, embedding_channels, H, W)
        x = torch.cat([images, labels], dim=1)                  # (B, in_channels + embedding_channels, H, W)

        x = self.initial(x)
        x = self.blocks(x)
        logits = self.final(x).view(-1, 1)
        return logits


class ConditionalGAN(nn.Module):
    def __init__(self, generator, discriminator, latent_dim=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discriminator = discriminator.to(self.device)
        self.generator = generator.to(self.device)

        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.latent_dim = latent_dim or generator.latent_dim
        self.global_epoch = 0

    def d_loss_fn(self, real_logits, fake_logits):
        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)
        d_real_loss = F.binary_cross_entropy_with_logits(real_logits, real_labels)
        d_fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
        d_loss = d_real_loss + d_fake_loss
        return d_loss, d_real_loss, d_fake_loss

    def g_loss_fn(self, fake_logits):
        real_labels = torch.ones_like(fake_logits)
        g_loss = F.binary_cross_entropy_with_logits(fake_logits, real_labels)
        return g_loss

    def train_step(self, batch):
        batch_size = batch["image"].size(0)
        labels = batch["label"].to(self.device)

        # (1) Update Discriminator
        real_images = batch["image"].to(self.device)
        real_logits = self.discriminator(real_images, labels)

        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(noises, labels).detach()
        fake_logits = self.discriminator(fake_images, labels)

        d_loss, d_real_loss, d_fake_loss = self.d_loss_fn(real_logits, fake_logits)

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # (2) Update Generator
        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(noises, labels)
        fake_logits = self.discriminator(fake_images, labels)
        g_loss = self.g_loss_fn(fake_logits)

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return dict(
            d_loss=d_loss.item(),
            real_loss=d_real_loss.item(),
            fake_loss=d_fake_loss.item(),
            g_loss=g_loss.item(),
            batch_size=batch_size,
        )
