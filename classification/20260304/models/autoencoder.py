import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure


class EncoderSmall(nn.Module):
    def __init__(self, latent_dim=2, in_channels=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc(x)
        return x    # latent

class DecoderSmall(nn.Module):
    def __init__(self, latent_dim=2, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )
        self.deconv3 = nn.ConvTranspose2d(32, out_channels,
            kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 4, 4)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = torch.sigmoid(x)    # nn.BCELoss()
        return x    # recon

class Encoder(nn.Module):
    def __init__(self, latent_dim=20, in_channels=3, num_features=64, image_size=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.num_features = num_features
        self.image_size = image_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.2)
        )

        self.feature_height = self.feature_width = image_size // 16
        self.flatten_size = num_features * self.feature_height * self.feature_width
        self.fc = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, self.flatten_size)
        latent = self.fc(x)
        return latent


class Decoder(nn.Module):
    def __init__(self, latent_dim=20, out_channels=3, num_features=64, image_size=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.num_features = num_features
        self.image_size = image_size

        self.feature_height = self.feature_width = image_size // 16
        self.flatten_size = num_features * self.feature_height * self.feature_width
        self.fc = nn.Linear(latent_dim, self.flatten_size)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(num_features, num_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.2)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(num_features, num_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.2)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(num_features, num_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.2)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(num_features, num_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.2)
        )
        self.deconv5 = nn.ConvTranspose2d(num_features, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.num_features, self.feature_height, self.feature_width)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = torch.sigmoid(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.BCELoss()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

    def forward(self, images):
        latent = self.encoder(images)
        recon = self.decoder(latent)
        return recon, latent

    def train_step(self, batch):
        images = batch["image"].to(self.device)
        recon, latent = self.forward(images)
        loss = self.loss_fn(recon, images)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ssim = self.ssim_metric(recon, images)
        return {
            "loss": loss.item(), 
            "ssim": ssim.item(), 
            "batch_size": images.size(0)
        }

    @torch.no_grad()
    def eval_step(self, batch):
        images = batch["image"].to(self.device)
        recon, latent = self.forward(images)
        loss = self.loss_fn(recon, images)
        ssim = self.ssim_metric(recon, images)
        return {
            "loss": loss.item(), 
            "ssim": ssim.item(), 
            "batch_size": images.size(0)
        }

