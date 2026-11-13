"""Редактирование изображения. Вся математика реализована самостоятельно."""
import numpy as np

def _saturate(arr):
    """Ручное ограничение значений диапазоном [0, 255] и приведение к uint8."""
    arr = arr.copy()
    arr[arr < 0] = 0
    arr[arr > 255] = 255
    return arr.astype(np.uint8)

def adjust_brightness(img, delta):
    """Яркость всех каналов: I' = I + delta."""
    return _saturate(img.astype(np.int16) + int(delta))

def adjust_channel(img, channel, delta):
    """Яркость одного канала: 0=R, 1=G, 2=B."""
    out = img.astype(np.int16).copy()
    out[:, :, channel] = out[:, :, channel] + int(delta)
    return _saturate(out)

def adjust_contrast(img, factor):
    """Контраст: I' = (I - 128) * factor + 128."""
    arr = img.astype(np.float32)
    return _saturate((arr - 128.0) * float(factor) + 128.0)

def negate(img, channel=None):
    """Негатив: I' = 255 - I.
    channel=None — все каналы, 0/1/2 — конкретный канал."""
    out = img.copy()
    if channel is None:
        out = 255 - out
    else:
        out[:, :, channel] = 255 - out[:, :, channel]
    return out

def swap_channels(img, c1, c2):
    """Обмен местами каналов c1 и c2."""
    out = img.copy()
    out[:, :, c1] = img[:, :, c2]
    out[:, :, c2] = img[:, :, c1]
    return out

def flip_horizontal(img):
    """Зеркальное отражение по горизонтали (лево-право)."""
    return img[:, ::-1].copy()

def flip_vertical(img):
    """Зеркальное отражение по вертикали (верх-низ)."""
    return img[::-1, :].copy()

def adjust_gamma(img, gamma):
    """Гамма-коррекция: I' = 255 * (I / 255) ** gamma."""
    arr = img.astype(np.float32) / 255.0
    arr = arr ** float(gamma)
    return _saturate(arr * 255.0)