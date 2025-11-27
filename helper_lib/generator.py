import math
import torch
from helper_lib.trainer import make_beta_schedule
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    _, C, H, W = imgs.shape

    grid = torch.ones(
        C,
        nrow * H + (nrow - 1) * pad,
        ncol * W + (ncol - 1) * pad,
        dtype=imgs.dtype
    )

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

def generate_diffusion_samples(
    model,
    device,
    num_samples: int = 16,
    T: int = 50,
    img_size: int = 32,
    seed: int | None = None,
    as_grid: bool = True,
    nrow: int | None = None,
    pad: int = 2,
):
    """
    Sample images from a DDPM-style diffusion model trained on CIFAR-10.

    model: UNet that predicts noise eps = model(x_t, t)
    Returns either:
      - a batch [N,3,H,W] in [0,1], or
      - a single grid [3,H_grid,W_grid] in [0,1] if as_grid=True
    """
    model = model.to(device).eval()

    if seed is not None:
        torch.manual_seed(seed)

    # schedule
    betas, alphas, alphas_cumprod = make_beta_schedule(T=T, device=device)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_ac = torch.sqrt(1.0 - alphas_cumprod)
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1, device=device), alphas_cumprod[:-1]],
        dim=0
    )

    # start from pure noise
    x = torch.randn(num_samples, 3, img_size, img_size, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            eps_theta = model(x, t_batch)

        beta_t = betas[t]
        sqrt_recip_alpha_t = sqrt_recip_alphas[t]
        sqrt_one_minus_ac_t = sqrt_one_minus_ac[t]
        ac_t = alphas_cumprod[t]
        ac_prev = alphas_cumprod_prev[t]

        # DDPM mean
        # mu_theta = 1/sqrt(alpha_t) * ( x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta )
        mu = (
            sqrt_recip_alpha_t
            * (x - beta_t / sqrt_one_minus_ac_t * eps_theta)
        )

        if t > 0:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(
                (1.0 - ac_prev) / (1.0 - ac_t) * beta_t
            )
            x = mu + sigma_t * noise
        else:
            x = mu

    # map from [-1,1] to [0,1]
    x = (x + 1) / 2.0
    x.clamp_(0.0, 1.0)

    if not as_grid:
        return x.cpu()

    if nrow is None:
        nrow = int(math.ceil(num_samples ** 0.5))
    ncol = nrow
    N, C, H, W = x.shape

    grid = torch.ones(
        C,
        nrow * H + (nrow - 1) * pad,
        ncol * W + (ncol - 1) * pad,
        dtype=x.dtype,
    )

    idx = 0
    for r in range(nrow):
        for c in range(ncol):
            if idx >= num_samples:
                break
            y0 = r * (H + pad)
            x0 = c * (W + pad)
            grid[:, y0:y0 + H, x0:x0 + W] = x[idx]
            idx += 1

    return grid.cpu()

def generate_energy_samples(
    model,
    device,
    num_samples: int = 16,
    img_size: int = 32,
    steps: int = 60,
    step_size: float = 0.1,
    noise_scale: float = 0.01,
    seed: int | None = None,
    as_grid: bool = True,
    nrow: int | None = None,
    pad: int = 2,
):
    """
    Sample from an energy-based model via simple Langevin dynamics:
      x_{k+1} = x_k - step_size * dE/dx + noise
    The model outputs scalar energy E(x), we move towards lower energy.
    """

    model = model.to(device).eval()

    if seed is not None:
        torch.manual_seed(seed)

    # initialize from Gaussian noise in [-1,1] roughly
    x = torch.randn(num_samples, 3, img_size, img_size, device=device)

    for _ in range(steps):
        x.requires_grad_(True)
        E = model(x).sum()
        grad, = torch.autograd.grad(E, x)
        # gradient descent on energy + noise
        x = x - step_size * grad
        if noise_scale > 0.0:
            x = x + noise_scale * torch.randn_like(x)
        x = x.detach().clamp_(-1.0, 1.0)

    # map from [-1,1] to [0,1]
    x = (x + 1) / 2.0
    x.clamp_(0.0, 1.0)

    if not as_grid:
        return x.cpu()

    if nrow is None:
        nrow = int(math.ceil(num_samples ** 0.5))
    ncol = nrow
    N, C, H, W = x.shape

    grid = torch.ones(
        C,
        nrow * H + (nrow - 1) * pad,
        ncol * W + (ncol - 1) * pad,
        dtype=x.dtype,
    )

    idx = 0
    for r in range(nrow):
        for c in range(ncol):
            if idx >= num_samples:
                break
            y0 = r * (H + pad)
            x0 = c * (W + pad)
            grid[:, y0:y0 + H, x0:x0 + W] = x[idx]
            idx += 1

    return grid.cpu()

##gpt2 text generation
_LLM_MODEL = None
_LLM_TOKENIZER = None


def load_finetuned_llm(
    model_dir: str = "checkpoints/gpt2_squad_finetuned",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    global _LLM_MODEL, _LLM_TOKENIZER

    if _LLM_MODEL is None or _LLM_TOKENIZER is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.to(device)
        _LLM_MODEL = model
        _LLM_TOKENIZER = tokenizer
        print(f"[Loaded LLM] {model_dir} on {device}")

    return _LLM_MODEL, _LLM_TOKENIZER


def generate_qa_answer(
    question: str,
    model_dir: str = "checkpoints/gpt2_squad_finetuned",
    max_length: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_finetuned_llm(model_dir=model_dir, device=device)

    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in text:
        answer_part = text.split("Answer:", 1)[1].strip()
    else:
        answer_part = text.strip()
        
    closing = "Let me know if you have any other questions."
    if closing in answer_part:
        before_closing = answer_part.split(closing, 1)[0].strip()
        answer_part = before_closing + " " + closing

    prefix = "That is a great question."
    if not answer_part.startswith(prefix):
        answer_part = f"{prefix} {answer_part}"

    return answer_part
