
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import io
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from PIL import Image


from helper_lib.model import get_model
from helper_lib.trainer import (
    train_gan,
    train_diffusion,   # 🔹 NEW: diffusion trainer
    train_energy,       # 🔹 NEW: energy trainer
    finetune_gpt2_on_squad # GPT-2 QA trainer
)
from helper_lib.generator import (
    generate_samples,            # GAN
    generate_diffusion_samples,  # NEW
    generate_energy_samples,      # NEW
    generate_qa_answer # GPT-2 QA
)

from helper_lib.data_loader import (
    get_mnist_loader,    # MNIST for GAN
    get_cifar10_loader  # 🔹 NEW: CIFAR-10 for diffusion / energy
)
from helper_lib.utils import save_model


# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------
app = FastAPI(title="HelperLib API", version="1.0.0")


# ---------------------------------------------------------
# Devices & global models
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GAN (MNIST)
GAN = get_model("GAN").to(DEVICE)
GAN_CHECKPOINT = "checkpoints/gan_mnist.pt"

# 🔹 NEW: Diffusion model (CIFAR-10)
DIFFUSION = get_model("DIFFUSION").to(DEVICE)
DIFFUSION_CHECKPOINT = "checkpoints/diffusion_cifar.pt"

# 🔹 NEW: Energy model (CIFAR-10)
ENERGY = get_model("ENERGY").to(DEVICE)
ENERGY_CHECKPOINT = "checkpoints/energy_cifar.pt"


# ---------------------------------------------------------
# Helper: try to load checkpoints
# ---------------------------------------------------------
def _try_load_weights(model, path: str, name: str):
    try:
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"[{name}] Loaded weights from {path}")
    except Exception as e:
        print(f"[{name}] No pre-trained weights loaded: {e}")


@app.on_event("startup")
def _startup():
    _try_load_weights(GAN, GAN_CHECKPOINT, "GAN")
    _try_load_weights(DIFFUSION, DIFFUSION_CHECKPOINT, "DIFFUSION")
    _try_load_weights(ENERGY, ENERGY_CHECKPOINT, "ENERGY")


# ---------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------
class TrainRequest(BaseModel):
    data_dir: str = Field("data", description="Path for MNIST dataset (will auto-download if missing)")
    batch_size: int = Field(128, description="Batch size for training")
    epochs: int = Field(5, description="Number of training epochs")
    lr: float = Field(0.0002, description="Learning rate for Adam optimizer")
    betas: List[float] = Field([0.5, 0.999], description="Betas for Adam optimizer (momentum terms)")
    z_dim: int = Field(100, description="Dimension of latent noise vector")
    save_path: str = Field("checkpoints/gan_mnist.pt", description="Path to save trained GAN model")

    class Config:
        schema_extra = {
            "example": {
                "data_dir": "data",
                "batch_size": 128,
                "epochs": 5,
                "lr": 0.0002,
                "betas": [0.5, 0.999],
                "z_dim": 100,
                "save_path": "checkpoints/gan_mnist.pt"
            }
        }


class GenerateRequest(BaseModel):
    num_samples: int = 36
    z_dim: int = 100
    nrow: int = 6
    normalize: bool = True


# 🔹 NEW: diffusion training schema (CIFAR-10)
class DiffusionTrainRequest(BaseModel):
    data_dir: str = Field("data_cifar", description="Path for CIFAR-10 dataset")
    batch_size: int = Field(128, description="Batch size for diffusion training")
    epochs: int = Field(20, description="Number of training epochs")
    lr: float = Field(0.0002, description="Learning rate for diffusion UNet")
    T: int = Field(1000, description="Number of diffusion timesteps")
    save_path: str = Field("checkpoints/diffusion_cifar.pt", description="Path to save diffusion model")

    class Config:
        schema_extra = {
            "example": {
                "data_dir": "data_cifar",
                "batch_size": 128,
                "epochs": 20,
                "lr": 0.0002,
                "T": 1000,
                "save_path": "checkpoints/diffusion_cifar.pt"
            }
        }


# 🔹 NEW: energy-model training schema (CIFAR-10)
class EnergyTrainRequest(BaseModel):
    data_dir: str = Field("data_cifar", description="Path for CIFAR-10 dataset")
    batch_size: int = Field(128, description="Batch size for energy model training")
    epochs: int = Field(10, description="Number of training epochs")
    lr: float = Field(0.0001, description="Learning rate for energy model")
    save_path: str = Field("checkpoints/energy_cifar.pt", description="Path to save energy model")

    class Config:
        schema_extra = {
            "example": {
                "data_dir": "data_cifar",
                "batch_size": 128,
                "epochs": 10,
                "lr": 0.0001,
                "save_path": "checkpoints/energy_cifar.pt"
            }
        }

class DiffusionGenerateRequest(BaseModel):
    num_samples: int = 16
    T: int = 50              # should match training T
    nrow: int = 4
    seed: Optional[int] = None


class EnergyGenerateRequest(BaseModel):
    num_samples: int = 16
    steps: int = 60
    step_size: float = 0.1
    noise_scale: float = 0.01
    nrow: int = 4
    seed: Optional[int] = None

class TrainResponse(BaseModel):
    detail: str

class QARequest(BaseModel):
    question: str


class QAResponse(BaseModel):
    answer: str

# ---------------------------------------------------------
# Utility: tensor -> PNG
# ---------------------------------------------------------
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

def grid_to_png(grid: torch.Tensor) -> bytes:
    grid = grid.clamp(0.0, 1.0)
    grid = (grid * 255).round().to(torch.uint8).cpu()

    if grid.size(0) == 1:
        arr = grid[0].numpy()
        pil = Image.fromarray(arr, mode="L")
    else:
        arr = grid.permute(1, 2, 0).numpy()
        pil = Image.fromarray(arr)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------
# Basic health endpoint
# ---------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "device": str(DEVICE)}


# ---------------------------------------------------------
# GAN endpoints (original code, kept)
# ---------------------------------------------------------
@app.post("/gan/train")
def gan_train(req: TrainRequest):
    train_loader = get_mnist_loader(req.data_dir, batch_size=req.batch_size, train=True)
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
            nrow=req.nrow
        )
    buf = io.BytesIO()
    Image.fromarray(imgs[0].numpy(), mode="L").save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# ---------------------------------------------------------
# 🔹 NEW: Diffusion training endpoint (CIFAR-10)
# ---------------------------------------------------------
@app.post("/diffusion/train")
def diffusion_train(req: DiffusionTrainRequest):
    train_loader = get_cifar10_loader(req.data_dir, batch_size=req.batch_size, train=True)
    train_diffusion(
        DIFFUSION,
        train_loader,
        device=DEVICE,
        T=req.T,
        epochs=req.epochs,
        lr=req.lr
    )
    save_model(DIFFUSION, req.save_path)
    return {
        "trained": True,
        "model": "DIFFUSION",
        "epochs": req.epochs,
        "saved_to": req.save_path
    }

@app.post("/diffusion/generate")
def diffusion_generate(req: DiffusionGenerateRequest):
    grid = generate_diffusion_samples(
        DIFFUSION,
        device=DEVICE,
        num_samples=req.num_samples,
        T=req.T,
        img_size=32,
        seed=req.seed,
        as_grid=True,
        nrow=req.nrow,
    )
    png_bytes = grid_to_png(grid)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")

# ---------------------------------------------------------
# 🔹 NEW: Energy-model training endpoint (CIFAR-10)
# ---------------------------------------------------------
@app.post("/energy/train")
def energy_train(req: EnergyTrainRequest):
    train_loader = get_cifar10_loader(req.data_dir, batch_size=req.batch_size, train=True)
    train_energy(
        ENERGY,
        train_loader,
        device=DEVICE,
        epochs=req.epochs,
        lr=req.lr
    )
    save_model(ENERGY, req.save_path)
    return {
        "trained": True,
        "model": "ENERGY",
        "epochs": req.epochs,
        "saved_to": req.save_path
    }

@app.post("/energy/generate")
def energy_generate(req: EnergyGenerateRequest):
    grid = generate_energy_samples(
        ENERGY,
        device=DEVICE,
        num_samples=req.num_samples,
        img_size=32,
        steps=req.steps,
        step_size=req.step_size,
        noise_scale=req.noise_scale,
        seed=req.seed,
        as_grid=True,
        nrow=req.nrow,
    )
    png_bytes = grid_to_png(grid)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")

@app.post("/train gpt-2", response_model=TrainResponse)
def train_model():
    # WARNING: this will block the request until training finishes
    finetune_gpt2_on_squad()
    return TrainResponse(detail="Training finished. Model saved to checkpoints/gpt2_squad_finetuned.")

@app.post("/answer gpt-2", response_model=QAResponse)
def answer(req: QARequest):
    answer_text = generate_qa_answer(req.question)
    return QAResponse(answer=answer_text)