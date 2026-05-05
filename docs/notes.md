## What train.py Did

1. **Downloaded the data** — grabbed 60,000 CIFAR-10 images from the internet and saved them to the `data/` folder

2. **Built the model** — created the neural network we defined in `model.py` with all its layers and weights starting at random values

3. **Ran the training loop** — for each of the 10 epochs it:
   - Fed the model 64 images at a time
   - Let the model guess what each image was
   - Measured how wrong the guesses were (loss)
   - Adjusted the weights to do better next time
   - Repeated for all 782 batches

4. **Evaluated after each epoch** — after every full pass through the training data it tested the model on the 10,000 test images it had never seen before

5. **Saved the best model** — every time the test accuracy improved it saved the weights to `models/cifar_model.pth`

So before `train.py` ran, our model was just an empty structure with random weights — essentially guessing randomly. After `train.py` finished, the weights have been tuned through 600,000+ adjustments and the model knows how to recognize images with 75.8% accuracy.

The file `models/cifar_model.pth` is the result — that's the trained brain we'll load for predictions.

## What evaluate.py Did

1. **Loaded the trained model** — read the weights from `models/cifar_model.pth` that train.py saved

2. **Ran the naive baseline** — tested what accuracy you get by always guessing "airplane" regardless of the image. Result: 10.0%

3. **Evaluated the neural network** — ran all 10,000 test images through the model and measured how many it got right. Result: 75.8%

4. **Broke down accuracy per class** — showed how well the model does on each individual category, revealing which classes it struggles with (cat: 49.8%) and which it excels at (automobile: 87.9%)

5. **Generated a visualization** — created `predictions.png` showing 18 test images with their predicted and actual labels. Green = correct, Red = wrong.

6. **Printed a summary** — compared baseline vs model side by side showing the +65.8% improvement

The key purpose of evaluate.py is honesty — it proves our model is genuinely learning and not just getting lucky, by comparing it against the dumbest possible strategy.