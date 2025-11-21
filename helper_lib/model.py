import torch
import torch.nn as nn
from torchvision.models import resnet18

def get_model(model_name):
    
    # FCNN
    if model_name == 'FCNN':
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    # CNN
    elif model_name == 'CNN':
        model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(64*12*12, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    elif model_name == 'EnhancedCNN':
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 10)

    elif model_name == 'VAE':

        class VAE(nn.Module):
            def __init__(self, latent_dim=20):
                super().__init__()
                self.flatten = nn.Flatten()
                self.encoder = nn.Sequential(
                    nn.Linear(28*28, 400),
                    nn.ReLU(),
                )
                self.fc_mu = nn.Linear(400, latent_dim)
                self.fc_logvar = nn.Linear(400, latent_dim)

                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 400),
                    nn.ReLU(),
                    nn.Linear(400, 28*28),
                    nn.Sigmoid()
                )
                self.unflatten = nn.Unflatten(1, (1, 28, 28))

            def reparameterize(self, mu, logvar):
                std = (0.5*logvar).exp()
                eps = std.new_empty(std.size()).normal_()
                return mu + eps * std

            def forward(self, x):
                x = self.flatten(x)
                h = self.encoder(x)
                mu = self.fc_mu(h)
                logvar = self.fc_logvar(h)
                z = self.reparameterize(mu, logvar)
                recon = self.decoder(z)
                recon = self.unflatten(recon)
                return recon, mu, logvar

        model = VAE()
        
    
    elif model_name == 'GAN':

        class Generator(nn.Module):
            def __init__(self, z_dim=100):
                super().__init__()
                self.fc = nn.Linear(z_dim, 128 * 7 * 7)
                self.net = nn.Sequential(
                    nn.BatchNorm2d(128),
                    # ConvTranspose2D: 128 → 64
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    # ConvTranspose2D: 64 → 1
                    nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
                    nn.Tanh()
                )

            def forward(self, z):
                x = self.fc(z)
                x = x.view(-1, 128, 7, 7)
                x = self.net(x)
                return x

        class Discriminator(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    # Conv2D: 1 → 64
                    nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    # Conv2D: 64 → 128
                    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Flatten(),
                    nn.Linear(128 * 7 * 7, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.net(x)

        class GAN(nn.Module):
            def __init__(self, z_dim=100):
                super().__init__()
                self.generator = Generator(z_dim)
                self.discriminator = Discriminator()

        model = GAN()
        
    elif model_name in ('DIFFUSION', 'DDPM'):

        # ---- Minimal UNet for 28x28 grayscale (noise-prediction for DDPM) ----
        class SinusoidalPosEmb(nn.Module):
            """t -> [B, emb_dim] sinusoidal time embedding"""
            def __init__(self, emb_dim=128):
                super().__init__()
                self.emb_dim = emb_dim

            def forward(self, t):
                # t: [B] or [B,1], values expected in [0, T-1] or normalized
                if t.dim() == 2 and t.size(1) == 1:
                    t = t.squeeze(1)
                device = t.device
                half = self.emb_dim // 2
                freqs = torch.exp(
                    torch.arange(half, device=device, dtype=torch.float32)
                    * (-torch.log(torch.tensor(10000.0)) / (half - 1))
                )
                args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
                emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
                if self.emb_dim % 2 == 1:
                    emb = torch.cat([emb, torch.zeros(len(t), 1, device=device)], dim=-1)
                return emb

        class ResidualBlock(nn.Module):
            def __init__(self, in_ch, out_ch, time_dim):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
                self.gn1 = nn.GroupNorm(8, out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
                self.gn2 = nn.GroupNorm(8, out_ch)
                self.act = nn.SiLU()
                self.time_proj = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_dim, out_ch)
                )
                self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

            def forward(self, x, t_emb):
                h = self.conv1(x)
                h = self.gn1(h)
                h = self.act(h)
                # add time
                time_h = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
                h = h + time_h
                h = self.conv2(h)
                h = self.gn2(h)
                h = self.act(h)
                return h + self.skip(x)

        class Down(nn.Module):
            def __init__(self, in_ch, out_ch, time_dim):
                super().__init__()
                self.block = ResidualBlock(in_ch, out_ch, time_dim)
                self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

            def forward(self, x, t_emb):
                x = self.block(x, t_emb)
                skip = x
                x = self.down(x)
                return x, skip

        class Up(nn.Module):
            def __init__(self, in_ch, out_ch, skip_ch, time_dim):
                super().__init__()
                # upsample: in_ch -> out_ch, spatial 2x
                self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
                # after concat, channels = out_ch + skip_ch
                self.block = ResidualBlock(out_ch + skip_ch, out_ch, time_dim)

            def forward(self, x, skip, t_emb):
                x = self.up(x)                      # [B, out_ch, H2, W2]
                x = torch.cat([x, skip], dim=1)     # [B, out_ch+skip_ch, H2, W2]
                x = self.block(x, t_emb)
                return x


        class SimpleUNet(nn.Module):
            def __init__(self, in_ch=3, base=64, time_dim=128):  # ⬅ make sure in_ch=3 for CIFAR
                super().__init__()
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(time_dim),
                    nn.Linear(time_dim, time_dim * 4),
                    nn.SiLU(),
                    nn.Linear(time_dim * 4, time_dim),
                )

                # encoder
                self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)          # 32x32
                self.down1 = Down(base, base * 2, time_dim)                  # 32->16, ch: 64 -> 128
                self.down2 = Down(base * 2, base * 4, time_dim)              # 16->8,  ch: 128 -> 256

                # bottleneck
                self.bot1 = ResidualBlock(base * 4, base * 4, time_dim)
                self.bot2 = ResidualBlock(base * 4, base * 4, time_dim)

                # decoder
                # skip s2 has channels base*4, skip s1 has channels base*2
                self.up1 = Up(base * 4, base * 2, skip_ch=base * 4, time_dim=time_dim)  # 8->16
                self.up2 = Up(base * 2, base,     skip_ch=base * 2, time_dim=time_dim)  # 16->32

                self.out_norm = nn.GroupNorm(8, base)
                self.out_act = nn.SiLU()
                self.out_conv = nn.Conv2d(base, in_ch, 3, padding=1)

            def forward(self, x, t):
                t_emb = self.time_emb(t)                 # [B, time_dim]
                x0 = self.in_conv(x)                     # [B, base, 28, 28]
                d1, s1 = self.down1(x0, t_emb)           # [B, 2b, 14, 14], skip s1
                d2, s2 = self.down2(d1, t_emb)           # [B, 4b, 7, 7],  skip s2
                b = self.bot2(self.bot1(d2, t_emb), t_emb)

                u1 = self.up1(b, s2, t_emb)              # [B, 2b, 14, 14]
                u2 = self.up2(u1, s1, t_emb)             # [B, b, 28, 28]
                h = self.out_act(self.out_norm(u2))
                eps = self.out_conv(h)                   # predict noise
                return eps

       # Instantiate a small UNet suitable for CIFAR10 32x32 RGB
        model = SimpleUNet(in_ch=3, base=64, time_dim=128)
        
    elif model_name == "ENERGY":

        class EnergyCNN(nn.Module):
            """
            Simple energy model for CIFAR10.
            Input:  x [B,3,32,32]
            Output: scalar energy E(x) [B]
            """
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(3, 64, 3, stride=2, padding=1),  # 32 -> 16
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1), # 16 -> 8
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),# 8 -> 4
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(256 * 4 * 4, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)   # energy scalar
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        model = EnergyCNN()

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model