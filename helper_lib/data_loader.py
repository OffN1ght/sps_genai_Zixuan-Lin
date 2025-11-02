import torch
from torchvision import datasets, transforms
def get_data_loader(data_dir, batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        transform=transform,
        download=True
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )

    return loader