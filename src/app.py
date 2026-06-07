import tkinter as tk
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw, ImageTk
from network import NeuralNetwork
from camera import FingerTracker

CANVAS_SIZE  = 560  # bigger = better camera but slower
BRUSH_RADIUS = 10   # NOTE: use 10 since MNIST lines are just too thin and 12 was too thick
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "..", "model", "model.npz")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("SketchMind")
        self.root.configure(bg="#1e1e1e")

        # load the trained model - weights were saved to model.npz by train.py
        self.net = NeuralNetwork([784, 128, 64, 10])
        self.net.load(MODEL_PATH)

        # PIL image is the actual drawing data that I send to the NN for prediction, it starts blank and I draw on it in on_draw() and update_camera()
        # the Tkinter canvas is just the visual layer the user sees
        self.image   = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.drawer  = ImageDraw.Draw(self.image)

        self.camera_mode  = False  # toggled by the camera button we start at the False = OFF
        self.tracker      = None   # FingerTracker instance it is linked to camera.py
        self.prev_pos     = None   # needed to draw lines between frames again in camera.py
        self.clear_timer  = None   # auto-clear variable, used in schedule_clear() and on_draw()

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

        self.canvas.bind("<B1-Motion>", self.on_draw)           # starts while holding left mouse button
        self.canvas.bind("<ButtonRelease-1>", self.on_release)  # starts when mouse button is released
        self.root.bind("<c>", lambda e: self.clear())           # if we press C we clear without clicking

    def on_draw(self, event):
        # called on every mouse move while drawing
        # draws on both the visible canvas and the PIL image
        r = BRUSH_RADIUS
        x, y = event.x, event.y
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.drawer.ellipse([x-r, y-r, x+r, y+r], fill=255)
        self.run_prediction()   # update prediction live while drawing
        self.schedule_clear()   # reset the auto-clear timer

    def setup_panel(self):
        # right side panel - we have there prediction label, confidence, probability bars, buttons...
        panel = tk.Frame(self.root, bg="#1e1e1e")
        panel.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        tk.Label(panel, text="Prediction", font=("Arial", 16, "bold"),
                 bg="#1e1e1e", fg="white").pack(pady=(0, 5))

        # big number showing what the network thinks was drawn // still working on how to make it more accurate and less jumpy while drawing
        self.prediction_label = tk.Label(
            panel, text="?", font=("Arial", 72, "bold"),
            bg="#1e1e1e", fg="#00ff88", width=3
        )
        self.prediction_label.pack()

        # shows the confidence percentage for example "67.5% sure"
        self.confidence_label = tk.Label(
            panel, text="draw something",
            font=("Arial", 12), bg="#1e1e1e", fg="#aaaaaa"
        )
        self.confidence_label.pack(pady=(0, 20))

        tk.Label(panel, text="Probabilities", font=("Arial", 11, "bold"),
                 bg="#1e1e1e", fg="white").pack(pady=(10, 4))

        # one bar per digit (0-9), width = probability, updated in run_prediction()
        self.bars = []
        self.bar_labels = []

        for digit in range(10):
            row = tk.Frame(panel, bg="#1e1e1e")
            row.pack(fill="x", pady=1)

            tk.Label(row, text=str(digit), width=2, bg="#1e1e1e",
                     fg="white", font=("Arial", 10)).pack(side="left")

            bar_bg = tk.Frame(row, bg="#333333", height=14, width=180)
            bar_bg.pack(side="left", padx=4)
            bar_bg.pack_propagate(False)

            bar = tk.Frame(bar_bg, bg="#00ff88", height=14, width=0)
            bar.place(x=0, y=0, relheight=1)

            pct = tk.Label(row, text="0%", width=4, bg="#1e1e1e",
                           fg="#aaaaaa", font=("Arial", 9))
            pct.pack(side="left")

            self.bars.append(bar)
            self.bar_labels.append(pct)

        # camera toggle button - starts/stops the webcam and finger tracking in camera.py
        self.camera_btn = tk.Button(
            panel, text="Use Camera", font=("Arial", 12),
            command=self.toggle_camera, bg="#444444", fg="white",
            relief="flat", padx=10, pady=5
        )
        self.camera_btn.pack(pady=(15, 5))

        tk.Button(
            panel, text="Clear", font=("Arial", 12),
            command=self.clear, bg="#444444", fg="white",
            relief="flat", padx=10, pady=5
        ).pack()

    # when the mouse button is released, we run one final prediction to update the panel with the finished drawing
    def on_release(self, event):
        # run one final prediction when the mouse button is released, just to correct nonsense while drawing
        self.run_prediction()

    def center_image(self, img):
        # MNIST digits are always centered and cropped to the bounding box of the strokes, so I do the same here
        # without this the network gets confused by empty space around the digit
        pixels = np.array(img)
        rows   = np.any(pixels > 0, axis=1)
        cols   = np.any(pixels > 0, axis=0)

        if not rows.any():
            return img  # nothing drawn yet, return as is

        # find the bounding box of the drawing
        top, bottom = np.where(rows)[0][[0, -1]]
        left, right = np.where(cols)[0][[0, -1]]

        cropped = img.crop((left, top, right + 1, bottom + 1))

        # scale it to fit inside a 160x160 area, then paste it centered in 200x200
        result = Image.new("L", (200, 200), 0)
        scale  = min(160 / cropped.width, 160 / cropped.height)
        new_w  = int(cropped.width * scale)
        new_h  = int(cropped.height * scale)
        cropped  = cropped.resize((new_w, new_h), Image.LANCZOS)
        offset_x = (200 - new_w) // 2
        offset_y = (200 - new_h) // 2
        result.paste(cropped, (offset_x, offset_y))
        return result

    def schedule_clear(self):
        # every stroke resets this timer
        # if 10 seconds pass without any drawing, clear() is called automatically
        if self.clear_timer:
            self.root.after_cancel(self.clear_timer)
        self.clear_timer = self.root.after(10000, self.clear)

    def run_prediction(self):
        # prepares the drawing and runs it through the NN to get the predicted digit and confidence, then updates the panel with the results
        # called from on_draw(), on_release(), and update_camera()
        centered = self.center_image(self.image)
        small    = centered.resize((28, 28), Image.LANCZOS)  # NN expects 28x28
        pixels   = np.array(small).flatten() / 255.0         # flatten to 784 values, scale to 0-1
        pixels   = pixels.reshape(1, 784)                    # NN expects shape (1, 784)

        digit, probs = self.net.predict(pixels)
        digit      = digit[0]
        confidence = probs[0][digit] * 100

        # update the big prediction label and confidence text
        self.prediction_label.config(text=str(digit))
        self.confidence_label.config(text=f"{confidence:.1f}% sure")

        # update all 10 probability bars
        for i in range(10):
            prob      = probs[0][i]
            bar_width = int(prob * 180)  # max bar width is 180px
            color     = "#00ff88" if i == digit else "#446644"
            self.bars[i].place(x=0, y=0, relheight=1, width=bar_width)
            self.bars[i].config(bg=color)
            self.bar_labels[i].config(text=f"{prob*100:.0f}%")

    def toggle_camera(self):
        # switches between mouse drawing and camera drawing
        self.camera_mode = not self.camera_mode
        if self.camera_mode:
            self.tracker = FingerTracker()
            self.camera_btn.config(text="Stop Camera", bg="#cc4444")
            self.update_camera()  # starts the camera loop
        else:
            self.stop_camera()
            self.camera_btn.config(text="Use Camera", bg="#444444")
            self.prev_pos = None

    def update_camera(self):
        # runs every 30ms while camera mode is active
        # gets the latest frame and finger position from FingerTracker
        if not self.camera_mode:
            return

        frame, pos, drawing = self.tracker.get_finger_pos()

        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img       = Image.fromarray(frame_rgb).resize((CANVAS_SIZE, CANVAS_SIZE))

            # I paste a white image using the stroke mask so strokes show up as white over the camera
            # self.image is grayscale - white where I drew, black everywhere else
            colored_strokes = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
            img.paste(colored_strokes, mask=self.image)

            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        if pos is not None:
            # convert finger position from camera coordinates to canvas coordinates
            cam_w = frame.shape[1]
            cam_h = frame.shape[0]
            x = int(pos[0] / cam_w * CANVAS_SIZE)
            y = int(pos[1] / cam_h * CANVAS_SIZE)

            if drawing and self.prev_pos is not None:
                r = BRUSH_RADIUS
                px, py = self.prev_pos
                # connect the previous finger position to the current one so the stroke stays continuous
                # finger tracking still jumps, so we also draw a dot at the current point for better coverage
                self.drawer.line([px, py, x, y], fill=255, width=r * 2)
                self.drawer.ellipse([x-r, y-r, x+r, y+r], fill=255)
                self.run_prediction()
                self.schedule_clear()

            # remember position only while drawing, reset to None when paused
            self.prev_pos = (x, y) if drawing else None
        else:
            self.prev_pos = None  # hand not visible, reset so next stroke starts fresh

        self.root.after(30, self.update_camera)  # schedule next frame

    def stop_camera(self):
        # releases the webcam and goes back to a plain black canvas
        if self.tracker:
            self.tracker.release()
            self.tracker = None
        self.canvas.delete("all")

    def clear(self):
        # wipes everything - canvas, PIL image, prediction panel...
        self.canvas.delete("all")
        self.image  = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.drawer = ImageDraw.Draw(self.image)
        self.clear_timer = None
        self.prediction_label.config(text="?")
        self.confidence_label.config(text="draw something")
        for i in range(10):
            self.bars[i].place(x=0, y=0, relheight=1, width=0)
            self.bar_labels[i].config(text="0%")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()