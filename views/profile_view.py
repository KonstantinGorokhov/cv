import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
)


class ProfileView:
    """Окно с графиком яркости вдоль выбранной строки изображения."""

    def __init__(self, root, gray):
        self.gray = gray
        self.win = tk.Toplevel(root)
        self.win.title("Профиль яркости строки")

        top = tk.Frame(self.win)
        top.pack(
            side=tk.TOP, fill=tk.X, padx=5, pady=5
        )
        tk.Label(top, text=f"Номер строки (0..{gray.shape[0] - 1}):").pack(
            side=tk.LEFT
        )
        self.row_var = tk.StringVar(
            value=str(gray.shape[0] // 2)
        )
        tk.Entry(top, textvariable=self.row_var, width=7).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(top, text="Показать", command=self.draw).pack(
            side=tk.LEFT
        )

        self.fig = Figure(figsize=(8, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(
            self.fig, master=self.win
        )
        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH, expand=True
        )
        self.draw()

    def draw(self):
        try:
            row = int(
                self.row_var.get()
            )
        except ValueError:
            messagebox.showerror(
                "Ошибка", "Номер строки должен быть числом"
            )
            return
        if (
            row < 0 or row >= self.gray.shape[0]
        ):
            messagebox.showerror(
                "Ошибка", "Такой строки нет в изображении"
            )
            return
        self.ax.clear()
        self.ax.plot(self.gray[row, :])
        self.ax.set_title(
            f"Профиль яркости, строка {row}"
        )
        self.ax.set_xlabel("Номер пикселя в строке")
        self.ax.set_ylabel("Яркость")
        self.ax.set_ylim(0, 255)
        self.canvas.draw_idle()
