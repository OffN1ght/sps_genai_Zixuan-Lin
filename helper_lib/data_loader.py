import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# def get_data_loader(data_dir, batch_size=32, train=True):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,)) 
#     ])

#     dataset = datasets.MNIST(
#         root=data_dir,
#         train=train,
#         transform=transform,
#         download=True
#     )

#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=train
#     )

#     return loader

def get_mnist_loader(data_dir, batch_size=64, train=True):
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def get_cifar10_loader(data_dir, batch_size=128, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # map pixels from [0,1] to roughly [-1,1]
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        transform=transform,
        download=True
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)