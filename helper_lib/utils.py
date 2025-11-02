import os
import torch

def save_model(model, path: str):
    directory = os.path.dirname(path)
    if directory: 
        os.makedirs(directory, exist_ok=True)

    torch.save(model.state_dict(), path)
    print(f"[Saved model] {os.path.abspath(path)}")


def load_model(model, path: str, map_location=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    print(f"[Loaded model] {os.path.abspath(path)}")
    return model