from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import io
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from PIL import Image

from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_samples
from helper_lib.data_loader import get_data_loader
from helper_lib.utils import save_model


app = FastAPI(title="HelperLib GAN API", version="1.0.0")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAN = get_model("GAN").to(DEVICE)

CHECKPOINT = "checkpoints/gan_mnist.pt"

def _try_load_weights(path: str = CHECKPOINT):
    try:
        state = torch.load(path, map_location=DEVICE)
        GAN.load_state_dict(state)
        print(f"[GAN] Loaded weights from {path}")
    except Exception as e:
        print(f"[GAN] No pre-trained weights loaded: {e}")

@app.on_event("startup")
def _startup():
    _try_load_weights()


class TrainRequest(BaseModel):
    data_dir: str = "data"
    batch_size: int = 128
    epochs: int = 5
    lr: float = 2e-4
    betas: Optional[List[float]] = None
    z_dim: int = 100
    save_path: str = CHECKPOINT

class GenerateRequest(BaseModel):
    num_samples: int = 36
    z_dim: int = 100
    nrow: int = 6
    normalize: bool = True


def tensor_to_png(tensor: torch.Tensor, nrow: int = 6, normalize: bool = True) -> bytes:
    grid = vutils.make_grid(tensor, nrow=nrow, normalize=normalize, padding=2) 
    
    if grid.size(0) == 1:
        arr = (grid[0] * 255).clamp(0, 255).byte().cpu().numpy()              
        pil = Image.fromarray(arr, mode="L")
    else:
        arr = (grid * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy() 
        pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@app.get("/")
def root():
    return {"status": "ok", "device": str(DEVICE)}

@app.post("/gan/train")
def gan_train(req: TrainRequest):
    train_loader = get_data_loader(req.data_dir, batch_size=req.batch_size, train=True)
    criterion = nn.BCELoss()
    betas = tuple(req.betas) if req.betas else (0.5, 0.999)
    optG = optim.Adam(GAN.generator.parameters(), lr=req.lr, betas=betas)
    optD = optim.Adam(GAN.discriminator.parameters(), lr=req.lr, betas=betas)

    train_gan(GAN, train_loader, criterion, (optG, optD),
              device=DEVICE, epochs=req.epochs, z_dim=req.z_dim)
    save_model(GAN, req.save_path)
    return {"trained": True, "epochs": req.epochs, "saved_to": req.save_path}

@app.post("/gan/generate")
def gan_generate(req: GenerateRequest):
    with torch.no_grad():
        imgs = generate_samples(
            GAN,
            device=DEVICE,
            num_samples=req.num_samples,
            z_dim=req.z_dim,
            as_grid=True, 
            nrow=req.nrow)
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(imgs[0].numpy(), mode="L").save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

