# evaluate.py — tests our trained model and compares it to a naive baseline
# This file's only job is honest evaluation.
# A model that can't beat a dumb baseline isn't worth using.

import torch
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend — saves to file instead of popup
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
import config
from dataset import get_dataloaders
from model import get_model


def evaluate_naive_baseline(test_loader):
    # The naive baseline: always predict class 0 (airplane) no matter what
    # This requires zero intelligence — it's the floor our model must beat
    # In CIFAR-10 each class has exactly 1000 test images out of 10,000 total
    # so always guessing "airplane" gets exactly 10% accuracy

    print("\n--- Naive Baseline (always guess 'airplane') ---")

    correct = 0
    total = 0

    for images, labels in test_loader:
        # Predict class 0 (airplane) for every single image
        predictions = torch.zeros(labels.size(0), dtype=torch.long)
        correct += predictions.eq(labels).sum().item()
        total += labels.size(0)

    accuracy = 100. * correct / total
    print(f"Baseline Accuracy: {accuracy:.1f}%")
    return accuracy


def evaluate_model(model, test_loader, device):
    # Evaluate our trained neural network on the test set

    print("\n--- Neural Network Evaluation ---")

    model.eval()  # disable dropout for evaluation
    correct = 0
    total = 0

    # Track per-class accuracy — tells us which categories the model struggles with
    class_correct = [0] * config.NUM_CLASSES
    class_total = [0] * config.NUM_CLASSES

    with torch.no_grad():  # no gradients needed — we're just measuring, not learning
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Track per-class results
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += predicted[i].eq(labels[i]).item()
                class_total[label] += 1

    overall_accuracy = 100. * correct / total
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")
    print("\nPer-class accuracy:")

    for i, cls in enumerate(config.CLASSES):
        class_acc = 100. * class_correct[i] / class_total[i]
        print(f"  {cls:<12} {class_acc:.1f}%")

    return overall_accuracy


def visualize_predictions(model, test_loader, device):
    # Show a grid of test images with their predicted and actual labels
    # Green label = correct prediction, Red label = wrong prediction

    print("\nGenerating prediction visualization...")

    model.eval()
    images_shown = 0
    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    axes = axes.flatten()

    with torch.no_grad():
        for images, labels in test_loader:
            images_device = images.to(device)
            outputs = model(images_device)
            _, predicted = outputs.max(1)

            for i in range(images.size(0)):
                if images_shown >= 18:  # show 18 images total
                    break

                # Convert tensor back to displayable image
                # Reverse the normalization we applied in dataset.py
                img = images[i].numpy().transpose(1, 2, 0)
                mean = np.array([0.4914, 0.4822, 0.4465])
                std = np.array([0.2023, 0.1994, 0.2010])
                img = std * img + mean  # undo normalization
                img = np.clip(img, 0, 1)  # keep values in valid range

                actual = config.CLASSES[labels[i].item()]
                pred = config.CLASSES[predicted[i].item()]
                is_correct = actual == pred

                axes[images_shown].imshow(img)
                axes[images_shown].axis('off')

                # Green if correct, red if wrong
                color = 'green' if is_correct else 'red'
                axes[images_shown].set_title(
                    f"Pred: {pred}\nActual: {actual}",
                    color=color,
                    fontsize=8
                )

                images_shown += 1

            if images_shown >= 18:
                break

    plt.suptitle('Model Predictions (Green=Correct, Red=Wrong)', fontsize=12)
    plt.tight_layout()

    # Save to file instead of showing in a popup
    output_path = os.path.join(config.BASE_DIR, 'predictions.png')
    plt.savefig(output_path)
    print(f"Visualization saved to: {output_path}")


def main():
    # Load data
    _, test_loader = get_dataloaders()

    # Load our trained model
    model, device = get_model()

    if not os.path.exists(config.MODEL_PATH):
        print("No trained model found. Please run train.py first.")
        return

    # Load the saved weights into the model
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    print("Loaded trained model.")

    # Run both evaluations
    baseline_acc = evaluate_naive_baseline(test_loader)
    model_acc = evaluate_model(model, test_loader, device)

    # Final comparison
    print("\n--- Summary ---")
    print(f"Naive baseline accuracy:  {baseline_acc:.1f}%")
    print(f"Neural network accuracy:  {model_acc:.1f}%")
    print(f"Improvement over baseline: +{model_acc - baseline_acc:.1f}%")

    # Visualize some predictions
    visualize_predictions(model, test_loader, device)


if __name__ == '__main__':
    main()
