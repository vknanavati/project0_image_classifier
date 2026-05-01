# config.py — the single source of truth for all settings in our project
# Instead of hardcoding numbers scattered across multiple files,
# we put them all here. If we want to change something, we change it once.

import os

# --- Paths ---
# os.path.dirname(__file__) means "the folder this file lives in" (i.e. src/)
# os.path.join(..., '..') means "go up one level" to project0_image_classifier/
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

# Where we'll save and load the trained model file
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'cifar_model.pth')

# --- Dataset ---
# CIFAR-10 has 10 categories — these are the labels in the order the dataset uses
# The order matters: index 0 = airplane, index 1 = automobile, etc.
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

NUM_CLASSES = len(CLASSES)  # 10

# --- Training ---
# How many images the model sees at once before updating its weights
# Analogy: instead of studying one flashcard at a time, you study 64 at once
# then adjust your understanding. Faster than one at a time, more stable than all at once.
BATCH_SIZE = 64

# How many times we loop through the entire dataset during training
# One epoch = the model has seen every training image once
NUM_EPOCHS = 10

# Learning rate — how big a step we take when adjusting weights
# Too high: the model overcorrects and never settles. Too low: training takes forever.
# 0.001 is a safe, commonly used starting point.
LEARNING_RATE = 0.001

# --- Image dimensions ---
# CIFAR-10 images are 32x32 pixels with 3 color channels (Red, Green, Blue)
IMAGE_SIZE = 32
NUM_CHANNELS = 3

# --- Flask API ---
HOST = '0.0.0.0'  # Accept connections from any network interface
PORT = 5000
