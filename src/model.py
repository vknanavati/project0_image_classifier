# model.py — defines the architecture of our neural network
# This file's only job is to describe the structure of the model.
# It knows nothing about the data or the API.

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(__file__))
import config

class SimpleCNN(nn.Module):
    # CNN stands for Convolutional Neural Network
    # It's the standard architecture for image recognition
    # Analogy: think of it like a series of filters on an Instagram photo
    # each filter highlights different features — edges, shapes, textures, patterns

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # --- Feature Extraction Layers ---
        # These layers learn to detect visual patterns in the image
        # Each Conv2d layer scans the image with a small window (3x3 pixels)
        # looking for a specific pattern, like a horizontal edge or a curve

        self.features = nn.Sequential(
            # First block — detects simple low-level features like edges
            # in: 3 color channels, out: 32 feature maps
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # stabilizes training, like keeping scores on the same scale
            nn.ReLU(),           # activation function — fires the neuron if the pattern is found
                                 # Analogy: a light switch — either the pattern is there or it isn't

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # shrinks the image by half (32x32 -> 16x16)
                                 # keeps only the strongest signal in each 2x2 block
                                 # Analogy: summarizing a paragraph into one sentence
            nn.Dropout(0.25),    # randomly turns off 25% of neurons during training
                                 # prevents the model from memorizing instead of learning

            # Second block — detects more complex patterns like corners and textures
            # in: 32 feature maps, out: 64 feature maps
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # shrinks again (16x16 -> 8x8)
            nn.Dropout(0.25),
        )

        # --- Classification Layers ---
        # These layers take the extracted features and decide which class it is
        # Analogy: a detective who has gathered all the clues now makes a final verdict

        self.classifier = nn.Sequential(
            # Flatten turns the 3D feature maps into a 1D list of numbers
            # 64 feature maps × 8×8 pixels = 4096 numbers
            nn.Flatten(),

            nn.Linear(64 * 8 * 8, 512),  # fully connected layer — every neuron talks to every other
            nn.ReLU(),
            nn.Dropout(0.5),             # drop 50% here — this layer is most prone to memorizing

            nn.Linear(512, config.NUM_CLASSES),  # final layer — one output per class (10 total)
            # No activation here — we'll apply softmax later during prediction
            # to convert raw scores into probabilities that sum to 100%
        )

    def forward(self, x):
        # forward() defines what happens when data flows through the network
        # x is a batch of images — shape: [batch_size, 3, 32, 32]
        x = self.features(x)      # extract visual features
        x = self.classifier(x)    # classify based on those features
        return x                  # returns raw scores (logits) for each class


def get_model():
    # Creates the model and moves it to the best available device
    # GPU (cuda) is much faster for training — uses thousands of cores in parallel
    # If no GPU, we fall back to CPU — slower but works fine
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SimpleCNN().to(device)
    return model, device
