# Project 0 — Study Guide: Image Classification

## What Problem We Solved
We taught a computer to look at an image and identify what's in it across
10 categories. Given a photo, our system returns a predicted class and a
confidence score.

## The Core ML Concept: Convolutional Neural Networks
A CNN processes images in two stages:
1. **Feature extraction** — convolutional layers scan the image looking for
edges, shapes, and patterns. Early layers detect simple things like edges,
later layers combine those into complex things like "that's a wheel."
2. **Classification** — fully connected layers take those extracted features
and decide which class they belong to.

Training works by repeating four steps thousands of times:
1. Make a guess (forward pass)
2. Measure how wrong it was (loss)
3. Figure out which weights caused the error (backpropagation)
4. Nudge those weights in the right direction (optimizer)

## Key Technical Decisions
- **CIFAR-10 dataset** — 60,000 images across 10 classes, small enough to
train on a laptop
- **Batch size 64** — the model sees 64 images at a time before updating
weights. Faster than one at a time, more stable than all at once.
- **Data augmentation** — random flips and crops during training make the
model more robust by artificially increasing variety
- **Dropout** — randomly disabling neurons during training prevents the model
from memorizing instead of learning
- **Adam optimizer** — automatically adjusts the learning rate for each
weight, more efficient than basic gradient descent
- **Saved best model** — we track test accuracy after every epoch and only
save when it improves

## Results
| | Accuracy |
|---|---|
| Naive baseline (always guess airplane) | 10.0% |
| Our neural network | 75.8% |
| Improvement | +65.8% |

## Per-Class Performance
| Class | Accuracy |
|---|---|
| Automobile | 87.9% |
| Truck | 87.1% |
| Ship | 86.1% |
| Frog | 86.0% |
| Airplane | 79.4% |
| Horse | 76.8% |
| Deer | 75.7% |
| Dog | 69.3% |
| Bird | 59.4% |
| Cat | 49.8% |

## Limitations We Discovered
- The model only knows 10 classes — it can't say "I don't know"
- Trained on tiny 32x32 images — struggles with real world photos
- Cats and dogs are hard because they look similar to each other

## The Most Important Thing to Remember
**Always benchmark against a naive baseline before claiming your model works.**
A model that can't beat "always guess the most common class" is useless,
no matter how sophisticated it looks. Our model beat the baseline by 65.8%
— that's what makes it worth using.

## Files We Built
| File | Job |
|---|---|
| config.py | Single source of truth for all settings |
| dataset.py | Download, transform, and load data |
| model.py | Define the neural network architecture |
| train.py | Run the training loop and save the best model |
| evaluate.py | Honest evaluation against naive baseline |
| predict.py | Load model and run inference on new images |
| app.py | Flask API wrapping the whole system |