# dataset.py — responsible for downloading, preparing, and loading our data
# This file's only job is data. It knows nothing about the model or the API.

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os

# Add the src/ folder to Python's path so we can import config.py
sys.path.append(os.path.dirname(__file__))
import config

def get_transforms():
    # transforms are a pipeline of operations applied to every image before
    # it goes into the model. Think of it like a photo prep station.

    train_transform = transforms.Compose([
        # Randomly flip the image horizontally during training
        # Analogy: showing a student a photo and its mirror image
        # teaches them the concept isn't tied to left/right orientation
        transforms.RandomHorizontalFlip(),

        # Randomly crop the image after padding it with 4 pixels of blank space
        # This teaches the model that the object can appear anywhere in the frame
        transforms.RandomCrop(32, padding=4),

        # Convert the image from a PIL image (what Pillow uses) to a PyTorch tensor
        # A tensor is just a multi-dimensional array of numbers — the format PyTorch needs
        transforms.ToTensor(),

        # Normalize pixel values from [0, 1] to roughly [-1, 1]
        # These specific numbers (mean and std) are pre-calculated for CIFAR-10
        # Normalization helps the model learn faster and more stably
        # Analogy: grading on a curve so all scores are on the same scale
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),  # average pixel value per channel
            std=(0.2023, 0.1994, 0.2010)    # spread of pixel values per channel
        ),
    ])

    # For test data we don't do random flips or crops — we want consistent evaluation
    # We only convert to tensor and normalize, same scale as training
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])

    return train_transform, test_transform


def get_dataloaders():
    train_transform, test_transform = get_transforms()

    # Download CIFAR-10 automatically if it's not already in our data/ folder
    # train=True gives us 50,000 training images
    train_dataset = datasets.CIFAR10(
        root=os.path.join(config.BASE_DIR, 'data'),
        train=True,
        download=True,
        transform=train_transform
    )

    # train=False gives us 10,000 test images — data the model never trains on
    test_dataset = datasets.CIFAR10(
        root=os.path.join(config.BASE_DIR, 'data'),
        train=False,
        download=True,
        transform=test_transform
    )

    # A DataLoader wraps a dataset and feeds it to the model in batches
    # Analogy: instead of handing exam papers one by one, you hand them in stacks of 64
    # shuffle=True means training data is randomly reordered each epoch
    # so the model doesn't memorize the order of examples
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False  # No need to shuffle test data — order doesn't matter here
    )

    return train_loader, test_loader
