def _flatten(window):
    """Развернуть окно в список значений (все каналы)"""
    values = []
    if window.ndim == 3:          # RGB
        for row in window:
            for pixel in row:
                for ch in pixel:
                    values.append(int(ch))
    else:                          # ч/б
        for row in window:
            for v in row:
                values.append(int(v))
    return values

def mean(values):
    """Среднее арифметическое."""
    if not values:
        return 0.0
    total = 0
    for v in values:
        total += v
    return total / len(values)

def std(values):
    """Стандартное отклонение."""
    if not values:
        return 0.0
    m = mean(values)
    total = 0.0
    for v in values:
        d = v - m
        total += d * d
    return (total / len(values)) ** 0.5

def get_window(img, x, y, size=11):
    """Квадратное окно size×size с центром в (x, y), с обрезкой по границам."""
    half = size // 2
    h, w = img.shape[:2]
    x0 = max(0, x - half)
    x1 = min(w, x + half + 1)
    y0 = max(0, y - half)
    y1 = min(h, y + half + 1)
    return img[y0:y1, x0:x1]

def window_stats(img, x, y, size=11):
    """Среднее и стандартное отклонение интенсивности окна size×size вокруг (x, y).
    Интенсивность каждого пикселя = (R + G + B) / 3."""
    window = get_window(img, x, y, size)
    values = []
    for row in window:
        for pixel in row:
            values.append(pixel_intensity(pixel))
    return mean(values), std(values)

def histogram(channel):
    """Гистограмма: список из 256 счётчиков (сколько пикселей с яркостью 0..255)."""
    counts = []
    for value in range(256):
        counts.append(int((channel == value).sum()))
    return counts

def pixel_intensity(rgb):
    """Интенсивность пикселя: [R + G + B] / 3."""
    return (int(rgb[0]) + int(rgb[1]) + int(rgb[2])) / 3