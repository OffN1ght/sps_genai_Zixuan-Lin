import math
import torch

def generate_samples(
    model,
    device,
    num_samples: int = 16,
    z_dim: int = 100,
    seed: int | None = None,
    as_grid: bool = False,
    nrow: int | None = None,
    pad: int = 2,
):

    G = model.generator.to(device).eval()

    if seed is not None:
        torch.manual_seed(seed)


    batch_cap = 256
    outs = []
    with torch.no_grad():
        for s in range(0, num_samples, batch_cap):
            b = min(batch_cap, num_samples - s)
            z = torch.randn(b, z_dim, device=device)
            x = G(z).to("cpu") 
            outs.append(x)
    imgs = torch.cat(outs, dim=0)


    imgs = (imgs + 1) / 2
    imgs.clamp_(0, 1)

    if not as_grid:
        return imgs
    if nrow is None:
        nrow = int(math.ceil(num_samples ** 0.5))
    ncol = nrow
    _, _, H, W = imgs.shape


    grid = torch.ones(1,
                      nrow * H + (nrow - 1) * pad,
                      ncol * W + (ncol - 1) * pad,
                      dtype=imgs.dtype)

    idx = 0
    for r in range(nrow):
        for c in range(ncol):
            if idx >= num_samples:
                break
            y0 = r * (H + pad)
            x0 = c * (W + pad)
            grid[:, y0:y0 + H, x0:x0 + W] = imgs[idx]
            idx += 1


    grid = (grid * 255).round().to(torch.uint8)
    return grid