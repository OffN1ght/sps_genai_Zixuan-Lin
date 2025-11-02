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


    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model