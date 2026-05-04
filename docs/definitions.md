## Understanding Training Output

`Epoch [1/10]` — We're on the 1st pass through the entire dataset out of 10 total. Think of it like re-reading a textbook 10 times — each pass the model gets a little better. one complete pass through the entire training dataset.

1 epoch contains all 782 batches — but remember each batch has 64 images, so:
782 batches × 64 images = 50,000 images = 1 epoch

`Batch [100/782]` — We're on batch 100 out of 782 total batches in this epoch. Remember we set `BATCH_SIZE = 64`, so 50,000 images ÷ 64 = 782 batches per epoch. The model sees 64 images at a time, updates its weights, then moves to the next 64.

`Loss: 2.243` — How wrong the model's guesses are right now. Lower is better. It starts high and should gradually decrease as training progresses. Think of it like a penalty score — we want to minimize it.Think of it like a golf score — lower is better, and zero would mean perfect predictions every time. In practice it never reaches zero.

`Accuracy: 22.1%` — How many images the model is correctly classifying so far in this epoch. It starts low and climbs. Remember our naive baseline is 10%, so we're already beating it — but we have a long way to go to hit our 70% target.

As training continues you should see the loss drop and accuracy climb with each epoch.