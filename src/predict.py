# predict.py — loads the trained model and makes predictions on new images
# This file's only job is inference — taking a single image and returning a prediction
# It's what the Flask API will call when someone uploads a photo

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(__file__))
import config
from model import get_model


def load_model():
    # Load the trained model from disk
    # We do this once when the API starts up, not on every request
    # Analogy: a doctor studying medicine once, then applying that knowledge
    # to each patient — not re-studying for every appointment

    model, device = get_model()

    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {config.MODEL_PATH}. "
            "Please run train.py first."
        )

    # Load the saved weights into the model structure
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model
