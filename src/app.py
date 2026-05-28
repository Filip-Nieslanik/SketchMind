import tkinter as tk
from PIL import Image, ImageDraw

CANVAS_SIZE = 280

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("SketchMind")
        self.root.configure(bg="#1e1e1e")

        # PIL image stores what is drawn so we can resize it to 28x28 later
        self.image  = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.drawer = ImageDraw.Draw(self.image)

        self.setup_canvas()

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

    def on_release(self, event):
        pass

    def clear(self):
        self.canvas.delete("all")
        self.image  = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.drawer = ImageDraw.Draw(self.image)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
