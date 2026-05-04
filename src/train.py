# train.py — runs the training loop
# This file's only job is to train the model and save it when done.
# Analogy: this is the study session. The model sees thousands of examples,
# makes guesses, gets corrected, and gradually improves.

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.dirname(__file__))
import config
from dataset import get_dataloaders
from model import get_model


def train():
    # Get our data loaders — training and test batches ready to go
    train_loader, test_loader = get_dataloaders()

    # Get our model and the device it's running on (CPU or GPU)
    model, device = get_model()

    # Loss function — measures how wrong the model's guess was
    # CrossEntropyLoss is standard for multi-class classification
    # Analogy: a judge scoring how far off your answer was from the right one
    criterion = nn.CrossEntropyLoss()

    # Optimizer — the algorithm that adjusts the model's weights after each batch
    # Adam is a popular choice — it's like a smart version of trial and error
    # it adjusts the learning rate automatically for each weight
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Learning rate scheduler — gradually reduces the learning rate over time
    # Analogy: when you're close to the answer, take smaller steps so you don't overshoot
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print("Starting training...")
    print(f"Training on {len(train_loader.dataset)} images")
    print(f"Testing on {len(test_loader.dataset)} images")
    print("-" * 50)

    best_accuracy = 0.0  # we'll save the best model we see during training

    for epoch in range(config.NUM_EPOCHS):
        # --- Training Phase ---
        model.train()  # puts model in training mode (enables dropout)
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to the same device as the model
            images, labels = images.to(device), labels.to(device)

            # Zero out gradients from the previous batch
            # Analogy: erasing the whiteboard before solving a new problem
            optimizer.zero_grad()

            # Forward pass — model makes predictions
            outputs = model(images)

            # Calculate how wrong the predictions were
            loss = criterion(outputs, labels)

            # Backward pass — figure out which weights caused the error
            # This is backpropagation — calculates gradients for every weight
            loss.backward()

            # Update the weights based on the gradients
            optimizer.step()

            # Track stats
            running_loss += loss.item()
            _, predicted = outputs.max(1)  # pick the class with highest score
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {running_loss/100:.3f} "
                      f"Accuracy: {100.*correct/total:.1f}%")
                running_loss = 0.0

        # --- Evaluation Phase ---
        # After each epoch, check how well the model does on data it never trained on
        model.eval()  # puts model in evaluation mode (disables dropout)
        test_correct = 0
        test_total = 0

        with torch.no_grad():  # don't calculate gradients during evaluation — saves memory
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_accuracy = 100. * test_correct / test_total
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}] "
              f"Test Accuracy: {test_accuracy:.1f}%")

        # Save the model if it's the best we've seen so far
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"New best model saved! Accuracy: {best_accuracy:.1f}%")

        # Step the learning rate scheduler
        scheduler.step()
        print("-" * 50)

    print(f"\nTraining complete! Best accuracy: {best_accuracy:.1f}%")


if __name__ == '__main__':
    train()
