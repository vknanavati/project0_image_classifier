# Project 0 — Image Classification Overview

## What We're Building
A system that looks at a photo and identifies what's in it. You'll be able to send an image to a Flask API and get back a response like `{"class": "dog", "confidence": 0.94}`.

## The Problem We're Solving
Computers don't "see" the way humans do. To a computer, an image is just a grid of numbers — each pixel is three numbers representing how much Red, Green, and Blue it contains. Our job is to build a system that takes those grids of numbers and learns to recognize patterns that correspond to categories like "cat" or "truck."

## Objectives
1. **Train a neural network** on 60,000 labeled images across 10 categories
2. **Benchmark honestly** — compare our model against the dumbest possible baseline (just always guessing the most common class)
3. **Visualize predictions** so we can actually see it working
4. **Wrap it in an API** so you can send any image and get a prediction back

## The 10 Categories
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Tools We're Using
| Tool | What it does |
|------|-------------|
| **PyTorch** | Builds and trains the neural network |
| **TorchVision** | Downloads CIFAR-10 dataset, handles image transforms |
| **Flask** | Wraps the model in an API |
| **Pillow** | Opens and processes image files |
| **NumPy** | Fast math on arrays of numbers |
| **Matplotlib** | Visualizes images and results |

## Files We're Building
src/
  config.py       ✅ done — all settings in one place
  dataset.py      — downloads and prepares the data
  model.py        — defines the neural network architecture
  train.py        — runs the training loop
  evaluate.py     — tests the model and compares to baseline
  predict.py      — loads the model and makes predictions
  app.py          — Flask API

## The ML Concept: How a Neural Network Learns
Think of it like a student taking a multiple choice test:

1. **Forward pass** — the network looks at an image and makes a guess
2. **Loss** — we measure how wrong the guess was (like a test score)
3. **Backpropagation** — we figure out which internal dials contributed to the wrong answer
4. **Weight update** — we nudge those dials in the right direction
5. **Repeat** — do this 60,000 images × 10 epochs = 600,000 times

After enough repetitions, the network gets good at guessing correctly.

## The Honest Baseline
Before claiming our model is good, we'll compare it against a **naive baseline** — a strategy that requires zero intelligence: just always predict the most common class. Since CIFAR-10 has 10 equal classes, always guessing "airplane" gives you 10% accuracy. Our neural network needs to meaningfully beat that to be worth anything.

## What Success Looks Like
| | Accuracy |
|---|---|
| Naive baseline (always guess "airplane") | ~10% |
| Our neural network (target) | 70%+ |

The Flask API will return predictions on new images you upload.