from tkinter import filedialog 

from views.main_view import (
    MainView,
)
from utils.image_io import (
    load_image,
    rgb_to_grayscale,
)
from utils import (
    adjust,
    filters,
)
from views.profile_view import (
    ProfileView,
)
from views.contrast_view import ContrastView
from views.lab4_view import Lab4View
import numpy as np
from PIL import (
    Image,
)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Статистика по изображению")

        self.original = None
        self.base = None
        self.current = None
        self.gray = None
        self.state = self._neutral_state()

        self.view = MainView(
            on_open=self.open_image,
            app=self
        )

    @staticmethod
    def _neutral_state():
        """Нейтральные (не изменяющие изображение) значения ползунков."""
        return {
            "brightness": {"all": 0, 0: 0, 1: 0, 2: 0},
            "contrast_factor": 1.0,
            "gamma": 1.0,
        }

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Images", "*.png *.bmp *.tif *.tiff"),
                ("All", "*.*"),
            ]
        )
        if not path:
            return
        self.original = load_image(path)
        self.current = self.original.copy()
        self.base = self.original.copy()
        self.state = self._neutral_state()
        self.view.reset_sliders()
        self._refresh_full()

    def _pipeline(self):
        """Пересчитать self.current из self.base по текущему self.state."""
        img = adjust.adjust_brightness(
            self.base, self.state["brightness"]["all"]
        )
        for ch in (0, 1, 2):
            delta = self.state["brightness"][ch]
            if delta:
                img = adjust.adjust_channel(img, ch, delta)
        img = adjust.adjust_contrast(
            img, self.state["contrast_factor"]
        )
        img = adjust.adjust_gamma(img, self.state["gamma"])
        self.current = img

    def _commit(self):
        """Зафиксировать текущий результат как новую базу и сбросить ползунки."""
        if self.current is None:
            return
        self.base = self.current.copy()
        self.state = (
            self._neutral_state()
        )
        self.view.reset_sliders()

    def _refresh_live(self):
        """Лёгкое обновление во время перетаскивания ползунка — только основное изображение."""
        self.view.show_image(self.current)

    def _refresh_full(self):
        """Полное обновление: пересчитать ч/б и обновить все панели (каналы, гистограммы)."""
        self.gray = rgb_to_grayscale(
            self.current
        )
        self.view.show_image(
            self.current
        )
        self.view.show_channels(
            self.current, self.gray
        )
        self.view.show_histograms(
            self.current, self.gray
        )

    def on_brightness(
        self, channel, value
    ):
        if self.current is None:
            return
        key = "all" if channel is None else channel
        self.state["brightness"][key] = value
        self._pipeline()
        self._refresh_live()

    def on_contrast(self, factor):
        if self.current is None:
            return
        self.state["contrast_factor"] = factor
        self._pipeline()
        self._refresh_live()

    def on_gamma(self, gamma):
        if self.current is None:
            return
        self.state["gamma"] = gamma
        self._pipeline()
        self._refresh_live()

    def on_slider_release(
        self,
    ):
        if self.current is None:
            return
        self._refresh_full()

    def on_channel_change(
        self, channel
    ):
        key = "all" if channel is None else channel
        self.view.set_brightness_value(
            self.state["brightness"][key]
        )

    def on_negate(self, channel):
        if self.current is None:
            return
        self.current = adjust.negate(
            self.current, channel
        )
        self._commit()
        self._refresh_full()

    def on_swap(self, c1, c2):
        if self.current is None:
            return
        self.current = adjust.swap_channels(
            self.current, c1, c2
        )
        self._commit()
        self._refresh_full()

    def on_flip(self, direction):
        if self.current is None:
            return
        if direction == "h":
            self.current = adjust.flip_horizontal(
                self.current
            )
        else:
            self.current = adjust.flip_vertical(self.current)
        self._commit()
        self._refresh_full()

    def on_reset(self):
        if self.original is None:
            return
        self.current = (
            self.original.copy()
        )
        self.base = self.original.copy()
        self.state = self._neutral_state()
        self.view.reset_sliders()
        self._refresh_full()

    def on_blur(self, connectivity):
        if self.current is None:
            return
        self.current = filters.blur(
            self.current, connectivity
        )
        self._commit()
        self._refresh_full()

    def on_save(self):
        if self.current is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tiff"),
            ],
        )
        if not path:
            return
        Image.fromarray(self.current).save(
            path
        )

    def on_profile(self):
        if self.current is None:
            return
        ProfileView(
            self.root, self.gray
        )

    def on_contrast_map(self):
        if self.current is None:
            return
        ContrastView(
            self.root, self.gray
        )

    # ---------- ЛР4 ----------

    def on_lab4(self):
        if self.current is None:
            return
        Lab4View(self)

    def apply_result(self, img):
        """Сделать результат из окна ЛР4 текущим изображением."""
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        self.current = img.copy()
        self._commit()
        self._refresh_full()
