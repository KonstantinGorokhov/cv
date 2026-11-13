"""Карта контрастности. Считается по яркости (ч/б) изображения."""
import numpy as np

def contrast_map(gray, mode, size=3):
    """Возвращает карту контрастности, значения растянуты на 0..255.
    mode:
    "4"- средняя разница яркости пикселя с 4 соседями
    "8"- средняя разница яркости пикселя с 8 соседями
    "win" - разброс яркости (max - min) в окне size x size
    """
    g = gray.astype(np.float32)
    h, w = g.shape
    if mode in ("4", "8"):
        if mode == "4":
            shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            shifts = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                      if (dy, dx) != (0, 0)]
        pad = np.pad(g, 1, mode="edge")
        total = np.zeros_like(g)
        for dy, dx in shifts:
            neighbor = pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            total += np.abs(g - neighbor)
        result = total / len(shifts)
    else:
        half = size // 2
        pad = np.pad(g, half, mode="edge")
        mx = g.copy()
        mn = g.copy()
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                part = pad[half + dy:half + dy + h, half + dx:half + dx + w]
                mx = np.maximum(mx, part)
                mn = np.minimum(mn, part)
        result = mx - mn
    top = result.max()
    if top > 0:
        result = result * 255.0 / top
    return result.astype(np.uint8)