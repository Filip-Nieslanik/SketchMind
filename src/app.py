import tkinter as tk
import numpy as np
import os
from PIL import Image, ImageDraw
from network import NeuralNetwork


CANVAS_SIZE = 280
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "model", "model.npz")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("SketchMind")
        self.root.configure(bg="#1e1e1e")

        # PIL image stores what is drawn so we can resize it to 28x28 later
        self.net = NeuralNetwork([784, 128, 64, 10])
        self.net.load(MODEL_PATH)

        self.image  = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.drawer = ImageDraw.Draw(self.image)

        self.setup_canvas()
        self.setup_panel()

    def setup_canvas(self):
        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            cursor="crosshair"
        )
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        self.canvas.bind("<B1-Motion>", self.on_draw)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_draw(self, event):
        r = 12
        x, y = event.x, event.y
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.drawer.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def setup_panel(self):
        panel = tk.Frame(self.root, bg="#1e1e1e")
        panel.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        tk.Label(panel, text="Prediction", font=("Arial", 16, "bold"),
                 bg="#1e1e1e", fg="white").pack(pady=(0, 5))

        self.prediction_label = tk.Label(
            panel, text="?", font=("Arial", 72, "bold"),
            bg="#1e1e1e", fg="#00ff88", width=3
        )
        self.prediction_label.pack()

        self.confidence_label = tk.Label(
            panel, text="draw something",
            font=("Arial", 12), bg="#1e1e1e", fg="#aaaaaa"
        )
        self.confidence_label.pack(pady=(0, 20))

        tk.Button(
            panel, text="Clear", font=("Arial", 12),
            command=self.clear, bg="#444444", fg="white",
            relief="flat", padx=10, pady=5
        ).pack()

    def on_release(self, event):
        # resize drawing to 28x28 and run it through the network
        small  = self.image.resize((28, 28), Image.LANCZOS)
        pixels = np.array(small).flatten() / 255.0
        pixels = pixels.reshape(1, 784)

        digit, probs = self.net.predict(pixels)
        digit      = digit[0]
        confidence = probs[0][digit] * 100

        self.prediction_label.config(text=str(digit))
        self.confidence_label.config(text=f"{confidence:.1f}% sure")

    def clear(self):
        self.canvas.delete("all")
        self.image  = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.drawer = ImageDraw.Draw(self.image)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
