import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils.contrast import contrast_map


class ContrastView:
    """Окно с картой контрастности. Формулу выбирает пользователь."""

    def __init__(self, root, gray):
        self.gray = gray
        self.win = tk.Toplevel(root)
        self.win.title("Карта контрастности")

        top = tk.Frame(self.win)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.mode = tk.StringVar(value="4")
        tk.Radiobutton(
            top, 
            text="По 4 соседям", 
            variable=self.mode,
            value="4"
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            top, 
            text="По 8 соседям", 
            variable=self.mode,
            value="8"
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            top, 
            text="По окну N×N, N =", 
            variable=self.mode,
            value="win"
        ).pack(side=tk.LEFT)
        self.size_var = tk.StringVar(value="5")
        tk.Entry(
            top, 
            textvariable=self.size_var, 
            width=4
        ).pack(side=tk.LEFT)
        tk.Button(
            top, 
            text="Построить", 
            command=self.draw
        ).pack(side=tk.LEFT, padx=10)

        self.fig = Figure(figsize=(7, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.draw()

    def draw(self):
        size = 3
        if self.mode.get() == "win":
            try:
                size = int(self.size_var.get())
            except ValueError:
                messagebox.showerror("Ошибка", "N должно быть числом")
                return
            if size < 3 or size % 2 == 0:
                messagebox.showerror("Ошибка", "N должно быть нечётным и не меньше 3")
                return
        result = contrast_map(self.gray, self.mode.get(), size)
        self.ax.clear()
        self.ax.imshow(result, cmap="gray", vmin=0, vmax=255)
        self.ax.set_title("Карта контрастности (светлее = контрастнее)")
        self.ax.axis("off")
        self.canvas.draw_idle()
