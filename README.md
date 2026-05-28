# SketchMind

School project — draw a digit (0-9) and a neural network tries to figure out what it is.

Built it without any ML libraries, just NumPy. Trained on MNIST.
You can draw with mouse or use webcam and draw with your finger.

---

## How it works

I wrote the neural network from scratch in NumPy — no PyTorch, no TensorFlow.
Trained it on MNIST (60k handwritten digits). Gets around 97% accuracy.

Network layers: 784 -> 128 -> 64 -> 10

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

Canvas clears itself after 2 seconds of no drawing.
