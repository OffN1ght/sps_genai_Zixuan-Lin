import torch
import torch.nn.functional as F
def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(data_loader):

            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                recon, mu, logvar = outputs
                recon_loss = criterion(recon, inputs)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / inputs.size(0)
                loss = recon_loss + kl_div
            else:
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
    return model

def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10, z_dim=100):
    
    G = model.generator.to(device)
    D = model.discriminator.to(device)

    if isinstance(optimizer, (list, tuple)):
        optG, optD = optimizer
    else:
        optG, optD = optimizer['G'], optimizer['D']

    for ep in range(epochs):
        running_d, running_g = 0.0, 0.0
        for real_imgs, _ in data_loader:
            real_imgs = real_imgs.to(device)
            bsz = real_imgs.size(0)
            real_lbl = torch.ones(bsz, 1, device=device)
            fake_lbl = torch.zeros(bsz, 1, device=device)

            # --- Train D ---
            with torch.no_grad():
                z = torch.randn(bsz, z_dim, device=device)
                fake_imgs = G(z)
            d_real = criterion(D(real_imgs), real_lbl)
            d_fake = criterion(D(fake_imgs.detach()), fake_lbl)
            d_loss = 0.5 * (d_real + d_fake)
            optD.zero_grad(); d_loss.backward(); optD.step()

            # --- Train G ---
            z = torch.randn(bsz, z_dim, device=device)
            gen_imgs = G(z)
            g_loss = criterion(D(gen_imgs), real_lbl)
            optG.zero_grad(); g_loss.backward(); optG.step()

            running_d += d_loss.item()
            running_g += g_loss.item()

        print(f"[GAN] Epoch {ep+1}/{epochs}  D: {running_d/len(data_loader):.4f}  G: {running_g/len(data_loader):.4f}")
        
    return model

# Diffusion model training utilities
def make_beta_schedule(T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod

def q_sample(x0, t, alphas_cumprod):
    noise = torch.randn_like(x0)
    ac = alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_ac = torch.sqrt(ac)
    sqrt_om = torch.sqrt(1.0 - ac)
    xt = sqrt_ac * x0 + sqrt_om * noise
    return xt, noise

def train_diffusion(model, data_loader, device="cpu", T=1000, epochs=10, lr=2e-4):
    model.to(device)
    model.train()

    betas, alphas, alphas_cumprod = make_beta_schedule(T=T, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for x, _ in data_loader:
            x = x.to(device)

            B = x.size(0)
            t = torch.randint(0, T, (B,), device=device, dtype=torch.long)

            xt, noise = q_sample(x, t, alphas_cumprod)
            pred_noise = model(xt, t)

            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        print(f"[Diffusion] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return model

# Energy-based model training utilities
def energy_loss(E_real, E_fake):
    return (F.softplus(E_real) + F.softplus(-E_fake)).mean()

def train_energy(model, data_loader, device="cpu", epochs=10, lr=1e-4):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for x, _ in data_loader:
            x = x.to(device)
            B = x.size(0)
            x_fake = torch.randn_like(x)

            E_real = model(x)
            E_fake = model(x_fake)

            loss = energy_loss(E_real, E_fake)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        print(f"[Energy] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return model
