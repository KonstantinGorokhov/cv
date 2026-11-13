import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from tkinter import ttk

from utils.stats import window_stats, pixel_intensity, histogram
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app import App


class MainView:
    def __init__(self, on_open, app: "App"):
        self.app = app
        self._suppress = False

        controls = tk.Frame(app.root)
        controls.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        tk.Button(
            controls, 
            text="Загрузить изображение", 
            command=on_open
        ).pack(side=tk.LEFT)
        tk.Button(
            controls, 
            text="Сохранить", 
            command=lambda: self.app.on_save()
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            controls, 
            text="Размытие (4 соседа)", 
            command=lambda: self.app.on_blur(4)
        ).pack(side=tk.LEFT)
        tk.Button(
            controls, 
            text="Размытие (8 соседей)", 
            command=lambda: self.app.on_blur(8)
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            controls, 
            text="Профиль строки", 
            command=lambda: self.app.on_profile()
        ).pack(side=tk.LEFT)
        tk.Button(
            controls, 
            text="Карта контрастности", 
            command=lambda: self.app.on_contrast_map()
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            controls,
            text="ЛР4: фильтры",
            command=lambda: self.app.on_lab4()
        ).pack(side=tk.LEFT)

        edit = tk.Frame(app.root)
        edit.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        tk.Label(edit, text="Канал:").pack(side=tk.LEFT)
        self.channel_var = tk.StringVar(value="Все")
        self.channel_combo = ttk.Combobox(
            edit, 
            textvariable=self.channel_var,
            values=["Все", "R", "G", "B"], 
            state="readonly", 
            width=5
        )
        self.channel_combo.pack(side=tk.LEFT)
        self.channel_combo.bind("<<ComboboxSelected>>", self._on_channel_change)

        tk.Label(edit, text="Яркость:").pack(side=tk.LEFT, padx=(10, 0))
        self.brightness_scale = tk.Scale(
            edit,
            from_=-100, 
            to=100,
            orient=tk.HORIZONTAL, 
            length=140,
            showvalue=False,
            command=self._on_brightness_live
        )
        self.brightness_scale.pack(side=tk.LEFT)
        self.brightness_scale.set(0)
        self.brightness_scale.bind(
            "<ButtonRelease-1>", 
            self._on_slider_release
        )

        tk.Label(edit, text="Контраст:").pack(side=tk.LEFT, padx=(10, 0))
        self.contrast_scale = tk.Scale(
            edit, 
            from_=0, 
            to=200,
            orient=tk.HORIZONTAL, 
            length=140,
            showvalue=True,
            command=self._on_contrast_live
        )
        self.contrast_scale.pack(side=tk.LEFT)
        self.contrast_scale.set(100)
        self.contrast_scale.bind(
            "<ButtonRelease-1>", 
            self._on_slider_release
        )

        tk.Label(edit, text="Гамма:").pack(side=tk.LEFT, padx=(10, 0))
        self.gamma_scale = tk.Scale(
            edit, 
            from_=20, 
            to=300,
            orient=tk.HORIZONTAL, 
            length=140,
            showvalue=False,
            command=self._on_gamma_live
        )
        self.gamma_scale.pack(side=tk.LEFT)
        self.gamma_scale.set(100)
        self.gamma_scale.bind(
            "<ButtonRelease-1>", 
            self._on_slider_release
        )

        tk.Button(
            edit, 
            text="Негатив", 
            command=self._on_negate
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            edit, 
            text="R↔G", 
            command=lambda: self._on_swap(0, 1)
        ).pack(side=tk.LEFT)
        tk.Button(
            edit, 
            text="R↔B", 
            command=lambda: self._on_swap(0, 2)
        ).pack(side=tk.LEFT, padx=(2, 0))
        tk.Button(
            edit, 
            text="G↔B", 
            command=lambda: self._on_swap(1, 2)
        ).pack(side=tk.LEFT, padx=(2, 0))
        tk.Button(
            edit, 
            text="⇄ Лево-Право", 
            command=lambda: self._on_flip("h")
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            edit, 
            text="⇅ Верх-Низ", 
            command=lambda: self._on_flip("v")
        ).pack(side=tk.LEFT)
        tk.Button(
            edit, 
            text="Сброс", 
            command=self._on_reset
        ).pack(side=tk.LEFT, padx=5)


        self.fig = Figure(figsize=(16, 9))
        gs = GridSpec(
            4, 3,
            figure=self.fig,
            width_ratios=[5, 1.5, 2.5],
            left=0.01, right=0.99, top=0.96, bottom=0.05,
            hspace=0.35,
            wspace=0.12,
        )

        self.ax_main = self.fig.add_subplot(gs[:, 0])
        self.ax_main.set_title("Основное изображение")
        self.ax_main.axis('off')

        self.ax_gray = self.fig.add_subplot(gs[0, 1])
        self.ax_gray.set_title("Ч/Б")
        self.ax_gray.axis('off')

        self.ax_r = self.fig.add_subplot(gs[1, 1])
        self.ax_r.set_title("R")
        self.ax_r.axis('off')

        self.ax_g = self.fig.add_subplot(gs[2, 1])
        self.ax_g.set_title("G")
        self.ax_g.axis('off')

        self.ax_b = self.fig.add_subplot(gs[3, 1])
        self.ax_b.set_title("B")
        self.ax_b.axis('off')

        self.ax_hist_gray = self.fig.add_subplot(gs[0, 2])
        self.ax_hist_gray.set_title("Hist. Ч/Б")

        self.ax_hist_r = self.fig.add_subplot(gs[1, 2])
        self.ax_hist_r.set_title("Hist. R")

        self.ax_hist_g = self.fig.add_subplot(gs[2, 2])
        self.ax_hist_g.set_title("Hist. G")

        self.ax_hist_b = self.fig.add_subplot(gs[3, 2])
        self.ax_hist_b.set_title("Hist. B")

        self.canvas = FigureCanvasTkAgg(self.fig, master=app.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


        self.current_image = None
        self.rect = None
        self.info_text = None

        self._pending_xy = None

        self._background = None

        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('draw_event', self._on_draw)

    def show_image(self, img):
        """Показать основное изображение."""
        self.current_image = img
        self.ax_main.clear()
        self.rect = None
        self.info_text = None
        self._background = None
        self.ax_main.imshow(img)
        self.ax_main.set_title("Основное изображение")
        self.ax_main.axis('off')
        self.canvas.draw_idle()

    def show_channels(self, img_rgb, img_gray):
        """Показать ч/б представление и три цветовых канала."""
        self.ax_gray.clear()
        self.ax_gray.imshow(img_gray, cmap='gray')
        self.ax_gray.set_title("Ч/Б")
        self.ax_gray.axis('off')

        self.ax_r.clear()
        self.ax_r.imshow(img_rgb[:, :, 0], cmap='gray')
        self.ax_r.set_title("R")
        self.ax_r.axis('off')

        self.ax_g.clear()
        self.ax_g.imshow(img_rgb[:, :, 1], cmap='gray')
        self.ax_g.set_title("G")
        self.ax_g.axis('off')

        self.ax_b.clear()
        self.ax_b.imshow(img_rgb[:, :, 2], cmap='gray')
        self.ax_b.set_title("B")
        self.ax_b.axis('off')

        self.canvas.draw_idle()

    def show_histograms(self, img_rgb, img_gray):
        """Показать гистограммы яркости ч/б и каждого канала."""
        items = [
            (self.ax_hist_gray, img_gray, 'gray', "Hist. Ч/Б"),
            (self.ax_hist_r, img_rgb[:, :, 0], 'red', "Hist. R"),
            (self.ax_hist_g, img_rgb[:, :, 1], 'green', "Hist. G"),
            (self.ax_hist_b, img_rgb[:, :, 2], 'blue', "Hist. B"),
        ] 
        for ax, channel, color, title in items:
            ax.clear()
            ax.bar(range(256), histogram(channel), width=1.0, color=color)
            ax.set_title(title)
            ax.set_xlim(0, 255)
            ax.set_ylim(bottom=0)

        self.canvas.draw_idle()

    def _on_draw(self, _event=None):
        """После полной перерисовки запоминаем фон без рамки и текста."""
        self._background = self.canvas.copy_from_bbox(self.fig.bbox)

    def on_mouse_move(self, event):
        """Запомнить позицию курсора и сразу обновить рамку и статистику."""
        if event.inaxes != self.ax_main: 
            return  
        if self.current_image is None: 
            return 
        self._pending_xy = (event.xdata, event.ydata) 
        self._apply_cursor_update()

    def _apply_cursor_update(self):
        """Обновить рамку 13×13 и статистику окна 11×11."""
        if self._pending_xy is None or self.current_image is None:
            return

        ex, ey = self._pending_xy
        x = int(round(ex))
        y = int(round(ey))

        h, w = self.current_image.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return

        if self.rect is None:
            self.rect = Rectangle(
                (x - 6.5, y - 6.5), 13, 13,
                linewidth=1.5, edgecolor='yellow', facecolor='none',
                animated=True
            )
            self.ax_main.add_patch(self.rect)
        else:
            self.rect.set_xy((x - 6.5, y - 6.5))
        self.rect.set_visible(True)

        rgb = self.current_image[y, x]
        intensity = pixel_intensity(rgb)

        mean_val, std_val = window_stats(self.current_image, x, y, size=11)

        info = (f"Пиксель: ({x}, {y})\n"
                f"RGB: ({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])})\n"
                f"Интенсивность: {intensity:.1f}\n"
                f"Окно 11×11: mean={mean_val:.1f}, std={std_val:.1f}")

        if self.info_text is None: 
            self.info_text = self.ax_main.text(
                0.98, 0.98, info,
                transform=self.ax_main.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                animated=True,
            )
        else: 
            self.info_text.set_text(info)

        if self._background is None:
            self.canvas.draw()
        self.canvas.restore_region(self._background)
        self.ax_main.draw_artist(self.rect)
        self.ax_main.draw_artist(self.info_text)
        self.canvas.blit(self.fig.bbox)

    def _selected_channel(self):
        return {"Все": None, "R": 0, "G": 1, "B": 2}[self.channel_var.get()]

    def reset_sliders(self):
        """Вернуть все три ползунка в нейтральное положение без вызова коллбэков."""
        self._suppress = True
        self.brightness_scale.set(0)
        self.contrast_scale.set(100)
        self.gamma_scale.set(100)
        self._suppress = False

    def set_brightness_value(self, value):
        """Показать на ползунке яркости сохранённое значение выбранного канала."""
        self._suppress = True
        self.brightness_scale.set(value)
        self._suppress = False

    def _on_brightness_live(self, value):
        if self._suppress:
            return
        self.app.on_brightness(self._selected_channel(), int(value))

    def _on_contrast_live(self, value):
        if self._suppress:
            return
        self.app.on_contrast(float(value) / 100.0)

    def _on_gamma_live(self, value):
        if self._suppress:
            return
        self.app.on_gamma(float(value) / 100.0)

    def _on_slider_release(self, _event=None):
        self.app.on_slider_release()

    def _on_channel_change(self, _event=None):
        if self._suppress:
            return
        self.app.on_channel_change(self._selected_channel())

    def _on_negate(self):
        self.app.on_negate(self._selected_channel())

    def _on_swap(self, c1, c2):
        self.app.on_swap(c1, c2)

    def _on_flip(self, direction):
        self.app.on_flip(direction)

    def _on_reset(self):
        self.app.on_reset()
