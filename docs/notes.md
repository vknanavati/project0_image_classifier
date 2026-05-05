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

## What config.py Does
1. **Defines all file paths** — sets the location of the project root and where to save/load the trained model (`models/cifar_model.pth`)
2. **Lists the 10 classes** — defines the CIFAR-10 category names in the exact order the dataset uses them. Order matters — index 0 = airplane, index 1 = automobile, etc.
3. **Sets training hyperparameters** — defines batch size (64), number of epochs (10), and learning rate (0.001) in one place
4. **Defines image dimensions** — sets image size (32x32) and number of color channels (3 for RGB)
5. **Configures the Flask API** — sets the host and port the server runs on

The key purpose of config.py is to be the single source of truth. Instead of magic numbers scattered across multiple files, everything is defined once here. If you want to train for 20 epochs instead of 10, you change it in one place and it updates everywhere.

---

## What dataset.py Does
1. **Defines two transform pipelines** — one for training (with random flips and crops for data augmentation) and one for testing (just normalize, no random changes)
2. **Downloads CIFAR-10 automatically** — fetches 50,000 training images and 10,000 test images from the internet and saves them to the `data/` folder
3. **Wraps data in DataLoaders** — packages the images into batches of 64 so the model can process them efficiently instead of one at a time
4. **Shuffles training data** — randomly reorders training images each epoch so the model doesn't memorize the order of examples

The key purpose of dataset.py is data preparation. The model never touches raw images — every image goes through the transform pipeline first to ensure it's in exactly the format the model expects.

---

## What model.py Does
1. **Defines the CNN architecture** — builds a two-block convolutional neural network with feature extraction layers and classification layers
2. **Feature extraction block** — two sets of convolutional layers that scan the image for edges, shapes, and patterns, with MaxPooling to shrink the image and Dropout to prevent memorization
3. **Classification block** — flattens the extracted features into a list of numbers, then uses fully connected layers to decide which of the 10 classes the image belongs to
4. **Adds BatchNorm and Dropout** — BatchNorm stabilizes training, Dropout randomly disables neurons to prevent the model from memorizing instead of learning
5. **Detects best available device** — automatically uses GPU if available, falls back to CPU if not

The key purpose of model.py is to define the structure of the neural network. It describes the architecture but contains no logic about training, data, or serving — just the model itself.


## What predict.py Does
1. **Loads the trained model from disk** — reads the saved weights once at startup, not on every prediction request
2. **Preprocesses incoming images** — resizes to 32x32, converts to RGB, applies the same normalization used during training so the model sees data in a familiar format
3. **Runs inference** — passes the image through the model in a single forward pass with no gradient calculation
4. **Converts scores to probabilities** — applies softmax to turn raw output scores into percentages that sum to 100%
5. **Returns a structured result** — gives back the predicted class, confidence score, and a full breakdown of probabilities for all 10 classes

The key purpose of predict.py is inference — taking a single new image and returning a prediction. It's the bridge between the trained model and the Flask API.

---

## What app.py Does
1. **Starts the Flask server** — creates the web server and loads the trained model once at startup
2. **Exposes a health check endpoint** — GET `/health` returns the model name and list of classes, useful for confirming the server is running
3. **Exposes a prediction endpoint** — POST `/predict` accepts an image file upload and returns a prediction
4. **Validates incoming requests** — checks that an image was actually sent, the filename isn't empty, and the file format is supported
5. **Handles the file temporarily** — saves the uploaded image to a temporary file, runs prediction, then deletes it
6. **Returns clean JSON** — sends back predicted class, confidence score, confidence percentage, and full class probabilities

The key purpose of app.py is serving — it's a thin wrapper around predict.py that handles the messy HTTP parts so the model is accessible to anything that can make a web request.