import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datasets import load_dataset
from transformers import AutoTokenizer

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

def get_squad_for_gpt2(
    model_name: str = "openai-community/gpt2",
    max_length: int = 256,
):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_datasets = load_dataset("rajpurkar/squad")
    raw_datasets["train"] = raw_datasets["train"].select(range(5000))
    
    def preprocess(example):
        question = example["question"]

        answers = example["answers"]["text"]
        answer = answers[0] if len(answers) > 0 else ""

        prompt = f"Question: {question}\nAnswer:"
        target = (
            f" That is a great question. {answer} "
            f"Let me know if you have any other questions."
        )
        full_text = prompt + target

        tokenized = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_datasets = raw_datasets.map(
        preprocess,
        remove_columns=raw_datasets["train"].column_names,
    )

    return tokenized_datasets, tokenizer