"""ЛР4: окно ВЧ-фильтрации, свёртки, выделения границ и углов."""
import time
import tkinter as tk
from typing import Callable
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from utils import filters_lab4 as f4


class Lab4View:
    def __init__(self, app):
        self.app = app
        self.result = None

        self.win = tk.Toplevel(app.root)
        self.win.title("ЛР4: ВЧ-фильтрация, свёртка, границы, углы")

        nb = ttk.Notebook(self.win)
        nb.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self._build_highpass(nb)
        self._build_convolution(nb)
        self._build_edges(nb)
        self._build_corners(nb)

        bottom = tk.Frame(self.win)
        bottom.pack(side=tk.TOP, fill=tk.X, padx=5)
        tk.Button(
            bottom,
            text="Применить результат к основному изображению",
            command=self._apply,
        ).pack(side=tk.LEFT)
        self.status = tk.Label(bottom, text="", anchor="w")
        self.status.pack(side=tk.LEFT, padx=10)

        self.fig = Figure(figsize=(14, 5))
        self.axes = [self.fig.add_subplot(1, 3, i + 1) for i in range(3)]
        for ax in self.axes:
            ax.axis("off")
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.05)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ---------- вкладки ----------

    def _build_highpass(self, nb):
        tab = tk.Frame(nb)
        nb.add(tab, text="High-pass")

        self.hp_method = tk.StringVar(value="box")
        tk.Label(tab, text="Размытие:").grid(row=0, column=0, sticky="w")
        tk.Radiobutton(tab, text="Усреднение", variable=self.hp_method, value="box").grid(row=0, column=1, sticky="w")
        tk.Radiobutton(tab, text="Гаусс", variable=self.hp_method, value="gauss").grid(row=0, column=2, sticky="w")

        tk.Label(tab, text="Размер n (нечётный):").grid(row=0, column=3, padx=(15, 0))
        self.hp_size = tk.StringVar(value="5")
        tk.Spinbox(tab, from_=3, to=51, increment=2, textvariable=self.hp_size, width=4).grid(row=0, column=4)

        tk.Label(tab, text="σ (для Гаусса):").grid(row=0, column=5, padx=(15, 0))
        self.hp_sigma = tk.StringVar(value="1.0")
        tk.Entry(tab, textvariable=self.hp_sigma, width=5).grid(row=0, column=6)

        tk.Label(tab, text="c:").grid(row=1, column=0, sticky="w")
        self.hp_c = tk.DoubleVar(value=1.0)
        tk.Scale(tab, from_=0.0, to=2.0, resolution=0.05, orient=tk.HORIZONTAL,
                 length=200, variable=self.hp_c).grid(row=1, column=1, columnspan=2, sticky="w")

        tk.Label(tab, text="Отрицательная яркость:").grid(row=1, column=3, padx=(15, 0))
        self.hp_neg = tk.StringVar(value=f4.NEG_ADD128)
        ttk.Combobox(tab, textvariable=self.hp_neg, values=f4.NEG_MODES,
                     state="readonly", width=20).grid(row=1, column=4, columnspan=3, sticky="w")

        tk.Button(tab, text="Вычислить ВЧ = ИСХ − РАЗМ·c", command=self._run_highpass).grid(
            row=0, column=7, rowspan=2, padx=15)

    def _build_convolution(self, nb):
        tab = tk.Frame(nb)
        nb.add(tab, text="Свёртка")

        tk.Label(tab, text="Стандартная матрица:").grid(row=0, column=0, sticky="w")
        self.conv_name = tk.StringVar(value="")
        combo = ttk.Combobox(tab, textvariable=self.conv_name,
                             values=list(f4.STANDARD_KERNELS), state="readonly", width=22)
        combo.grid(row=0, column=1, sticky="w")
        combo.bind("<<ComboboxSelected>>", self._on_standard_kernel)

        tk.Label(tab, text="Матрица n×n\n(через пробел,\nзапятую или\nпо строкам):",
                 justify="left").grid(row=1, column=0, sticky="nw")
        self.conv_text = tk.Text(tab, width=30, height=6, font=("Menlo", 11))
        self.conv_text.grid(row=1, column=1, rowspan=3, sticky="w", pady=2)

        self.conv_norm = tk.BooleanVar(value=True)
        self.conv_abs = tk.BooleanVar(value=False)
        self.conv_128 = tk.BooleanVar(value=False)
        tk.Checkbutton(tab, text="Нормализация (деление на сумму элементов)",
                       variable=self.conv_norm).grid(row=1, column=2, sticky="w", padx=10)
        tk.Checkbutton(tab, text="+128", variable=self.conv_128).grid(row=2, column=2, sticky="w", padx=10)
        tk.Checkbutton(tab, text="По модулю", variable=self.conv_abs).grid(row=3, column=2, sticky="w", padx=10)

        tk.Button(tab, text="Применить свёртку", command=self._run_convolution).grid(
            row=0, column=3, rowspan=2, padx=15)

        combo.set("Гаусс 3×3")
        self._on_standard_kernel()

    def _build_edges(self, nb):
        tab = tk.Frame(nb)
        nb.add(tab, text="Границы")

        tk.Label(tab, text="Оператор:").grid(row=0, column=0, sticky="w")
        self.ed_op = tk.StringVar(value="Sobel")
        tk.Radiobutton(tab, text="Собель", variable=self.ed_op, value="Sobel").grid(row=0, column=1, sticky="w")
        tk.Radiobutton(tab, text="Превитт", variable=self.ed_op, value="Prewitt").grid(row=0, column=2, sticky="w")

        tk.Label(tab, text="σ предварительного размытия (0 = нет):").grid(row=0, column=3, padx=(15, 0))
        self.ed_sigma = tk.StringVar(value="1.0")
        tk.Entry(tab, textvariable=self.ed_sigma, width=5).grid(row=0, column=4)

        tk.Label(tab, text="Порог (0 = без порога):").grid(row=1, column=0, sticky="w")
        self.ed_thr = tk.IntVar(value=50)
        tk.Scale(tab, from_=0, to=255, orient=tk.HORIZONTAL, length=250,
                 variable=self.ed_thr).grid(row=1, column=1, columnspan=3, sticky="w")

        tk.Button(tab, text="Выделить границы", command=self._run_edges).grid(
            row=0, column=5, rowspan=2, padx=15)

    def _build_corners(self, nb):
        tab = tk.Frame(nb)
        nb.add(tab, text="Углы (Харрис)")

        self.hr_k = tk.StringVar(value="0.04")
        self.hr_sigma = tk.StringVar(value="1.5")
        self.hr_thr = tk.StringVar(value="0.01")
        self.hr_nms = tk.StringVar(value="4")
        fields = [
            ("k (0.04–0.06):", self.hr_k),
            ("σ окна:", self.hr_sigma),
            ("Порог (доля от max R):", self.hr_thr),
            ("Радиус подавления немаксимумов:", self.hr_nms),
        ]
        for i, (label, var) in enumerate(fields):
            tk.Label(tab, text=label).grid(row=i // 2, column=(i % 2) * 2, sticky="w", padx=(0 if i % 2 == 0 else 15, 0))
            tk.Entry(tab, textvariable=var, width=6).grid(row=i // 2, column=(i % 2) * 2 + 1, sticky="w")

        tk.Button(tab, text="Найти углы", command=self._run_corners).grid(
            row=0, column=4, rowspan=2, padx=15)

    # ---------- обработчики ----------

    def _source(self):
        if self.app.current is None:
            messagebox.showerror("Ошибка", "Сначала загрузите изображение", parent=self.win)
            return None
        return self.app.current

    @staticmethod
    def _num(var, name, cast: Callable[[str], float] = float, minimum=None, odd=False):
        try:
            value = cast(var.get())
        except (ValueError, tk.TclError):
            raise ValueError(f"{name}: нужно число")
        if minimum is not None and value < minimum:
            raise ValueError(f"{name}: должно быть не меньше {minimum}")
        if odd and value % 2 == 0:
            raise ValueError(f"{name}: должно быть нечётным")
        return value

    def _run(self, compute):
        """Запустить вычисление, показать ошибку или время работы."""
        img = self._source()
        if img is None:
            return
        try:
            start = time.perf_counter()
            info = compute(img)
            elapsed = time.perf_counter() - start
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e), parent=self.win)
            return
        text = f"Время: {elapsed * 1000:.0f} мс"
        if info:
            text += f"   |   {info}"
        self.status.config(text=text)

    def _run_highpass(self):
        def compute(img):
            n = self._num(self.hp_size, "Размер n", int, 3, odd=True)
            sigma = self._num(self.hp_sigma, "σ", float)
            if sigma <= 0:
                raise ValueError("σ: должно быть больше 0")
            c = float(self.hp_c.get())
            blurred, hf = f4.high_pass(img, self.hp_method.get(), n, sigma, c, self.hp_neg.get())
            name = "усреднение" if self.hp_method.get() == "box" else f"Гаусс σ={sigma:g}"
            self._show(img, blurred, hf, f"РАЗМ ({name}, {n}×{n})", f"ВЧ = ИСХ − РАЗМ·{c:g} ({self.hp_neg.get()})")
            return None
        self._run(compute)

    def _on_standard_kernel(self, _event=None):
        name = self.conv_name.get()
        if name not in f4.STANDARD_KERNELS:
            return
        kernel, norm, use_abs, add128 = f4.STANDARD_KERNELS[name]
        self.conv_text.delete("1.0", tk.END)
        self.conv_text.insert("1.0", f4.format_kernel(kernel))
        self.conv_norm.set(norm)
        self.conv_abs.set(use_abs)
        self.conv_128.set(add128)

    def _run_convolution(self):
        def compute(img):
            kernel = f4.parse_kernel(self.conv_text.get("1.0", tk.END))
            res, divisor = f4.apply_kernel(
                img, kernel, self.conv_norm.get(), self.conv_abs.get(), self.conv_128.get()
            )
            n = len(kernel)
            self._show(img, None, res, "", f"Свёртка {n}×{n}")
            info = f"Матрица {n}×{n}"
            if self.conv_norm.get():
                info += f", делитель = {divisor:g}"
                if divisor == 1.0 and abs(f4.kernel_sum(kernel)) <= 1e-9:
                    info += " (сумма элементов = 0, деление не выполняется)"
            return info
        self._run(compute)

    def _run_edges(self):
        def compute(img):
            sigma = self._num(self.ed_sigma, "σ", float, 0)
            thr = int(self.ed_thr.get())
            mag, binary = f4.edges(img, self.ed_op.get(), sigma, thr)
            op = "Собель" if self.ed_op.get() == "Sobel" else "Превитт"
            if binary is None:
                self._show(img, None, mag, "", f"|∇I| ({op})")
            else:
                self._show(img, mag, binary, f"|∇I| ({op})", f"Границы: |∇I| ≥ {thr}")
            return None
        self._run(compute)

    def _run_corners(self):
        def compute(img):
            k = self._num(self.hr_k, "k", float, 0)
            sigma = self._num(self.hr_sigma, "σ окна", float)
            if sigma <= 0:
                raise ValueError("σ окна: должно быть больше 0")
            thr = self._num(self.hr_thr, "Порог", float, 0)
            nms = self._num(self.hr_nms, "Радиус", int, 1)
            ys, xs, r = f4.harris(img, k, sigma, thr, nms)
            marked = f4.draw_crosses(img, ys, xs)
            self._show(img, f4.response_to_image(r), marked, "Отклик Харриса R",
                       f"Найдено углов: {len(xs)}")
            return f"Углов: {len(xs)}"
        self._run(compute)

    # ---------- отображение ----------

    def _show(self, src, mid, res, mid_title, res_title):
        self.result = res
        items = [(src, "Исходное"), (mid, mid_title), (res, res_title)]
        for ax, (img, title) in zip(self.axes, items):
            ax.clear()
            ax.axis("off")
            if img is None:
                continue
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(img)
            ax.set_title(title, fontsize=10)
        self.canvas.draw_idle()

    def _apply(self):
        if self.result is None:
            messagebox.showinfo("ЛР4", "Сначала получите результат", parent=self.win)
            return
        self.app.apply_result(self.result)
