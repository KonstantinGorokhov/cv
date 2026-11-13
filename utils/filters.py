"""Размытие изображения (удаление шума). Математика своя."""
import numpy as np

def blur(img, connectivity):
    """Заменяет пиксель средним значением его и соседей.
    Connectivity = 4 (верх, низ, лево, право) или 8 (ещё и диагонали).
    На краях берём ближайший пиксель изображения."""
    h, w = img.shape[:2]
    arr = img.astype(np.float32)
    pad = np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode="edge")
    if connectivity == 4:
        shifts = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        shifts = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    total = np.zeros_like(arr)
    for dy, dx in shifts:
        total += pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    total = total / len(shifts)
    return (total + 0.5).astype(np.uint8)