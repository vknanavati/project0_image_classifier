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
    model, device = get_model()

    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {config.MODEL_PATH}. "
            "Please run train.py first."
        )

    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded from {config.MODEL_PATH}")
    return model, device


def predict(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])

    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(1)
        predicted_class = config.CLASSES[predicted_idx.item()]
        confidence_score = confidence.item()

        all_probabilities = {
            config.CLASSES[i]: round(probabilities[0][i].item(), 4)
            for i in range(config.NUM_CLASSES)
        }

    return {
        'predicted_class': predicted_class,
        'confidence': round(confidence_score, 4),
        'all_probabilities': all_probabilities
    }
