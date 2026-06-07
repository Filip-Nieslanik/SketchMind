# SketchMind

School project — draw a digit (0-9) and a neural network tries to figure out what it is.

Neural network written from scratch in NumPy, trained on MNIST.
You can draw with mouse or use webcam and draw with your finger.

---

## How it works

The network has 4 layers: 784 -> 128 -> 64 -> 10.
Trained on 60k handwritten digits, gets around 97% accuracy on the test set.

---

## Files

```
src/
  network.py   - the neural network (forward, backprop, save/load)
  train.py     - downloads MNIST and trains the model
  app.py       - the drawing app
  camera.py    - webcam + finger tracking (MediaPipe)

tests/
  test_network.py  - tests for the network
  test_train.py    - tests for data loading
```

---

## How to run

Install dependencies:
```bash
pip install -r requirements.txt
```

Train the model (downloads MNIST on first run):
```bash
python src/train.py
```

Run the app:
```bash
python src/app.py
```

---

## Camera mode

Click **Use Camera** and draw with your index finger.
Raise your middle finger too if you just want to move without drawing.

Canvas clears itself after 10 seconds of no drawing.
