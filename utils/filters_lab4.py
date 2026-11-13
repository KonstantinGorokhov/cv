import math
import numpy as np


def _edge_indices(n, before, after):
    return [min(max(i, 0), n - 1) for i in range(-before, n + after)]


def pad_edge(arr, pad_y, pad_x):
    h, w = arr.shape[:2]
    rows = _edge_indices(h, pad_y[0], pad_y[1])
    cols = _edge_indices(w, pad_x[0], pad_x[1])
    return arr[rows][:, cols]


def to_uint8(arr):
    out = arr.astype(np.float32)
    out[out < 0] = 0
    out[out > 255] = 255
    return (out + 0.5).astype(np.uint8)


def grayscale(img) -> np.ndarray:
    arr = img.astype(np.float32)
    if arr.ndim == 2:
        return arr
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def stretch(arr):
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-12:
        return np.zeros(arr.shape, dtype=np.float32)
    return (arr - mn) * (255.0 / (mx - mn))


def convolve(arr: np.ndarray, kernel) -> np.ndarray:
    arr = arr.astype(np.float32)
    kh = len(kernel)
    kw = len(kernel[0])
    h, w = arr.shape[:2]
    cy, cx = kh // 2, kw // 2
    pad = pad_edge(arr, (cy, kh - 1 - cy), (cx, kw - 1 - cx))
    out = np.zeros(arr.shape, dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            k = float(kernel[kh - 1 - i][kw - 1 - j])
            if k == 0.0:
                continue
            out += k * pad[i:i + h, j:j + w]
    return out


def convolve_separable(arr: np.ndarray, kernel_1d) -> np.ndarray:
    tmp = convolve(arr, [list(kernel_1d)])
    return convolve(tmp, [[v] for v in kernel_1d])


def kernel_sum(kernel):
    total = 0.0
    for row in kernel:
        for v in row:
            total += v
    return total


def apply_kernel(img, kernel, normalize, use_abs, add128):
    divisor = 1.0
    if normalize:
        s = kernel_sum(kernel)
        if abs(s) > 1e-9:
            divisor = s
    k = [[v / divisor for v in row] for row in kernel]
    res = convolve(img, k)
    if use_abs:
        res = np.abs(res)
    if add128:
        res = res + 128.0
    return to_uint8(res), divisor


def parse_kernel(text):
    for sep in (",", ";", "\t", "\n"):
        text = text.replace(sep, " ")
    tokens = [t for t in text.split(" ") if t]
    if not tokens:
        raise ValueError("Матрица пустая")
    values = []
    for t in tokens:
        try:
            values.append(float(t))
        except ValueError:
            raise ValueError(f"«{t}» — не число")
    n = math.isqrt(len(values))
    if n * n != len(values):
        raise ValueError(
            f"Введено {len(values)} чисел — это не квадрат n×n"
        )
    return [values[r * n:(r + 1) * n] for r in range(n)]


def format_kernel(kernel):
    lines = []
    for row in kernel:
        lines.append(" ".join(f"{v:g}" for v in row))
    return "\n".join(lines)


def box_kernel_1d(n):
    return [1.0 / n] * n


def gauss_kernel_1d(n, sigma):
    half = n // 2
    vals = [math.exp(-(i * i) / (2.0 * sigma * sigma)) for i in range(-half, half + 1)]
    s = sum(vals)
    return [v / s for v in vals]


def gauss_size(sigma):
    return 2 * int(math.ceil(3.0 * sigma)) + 1


def _outer(v):
    return [[a * b for b in v] for a in v]


SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

STANDARD_KERNELS = {
    "Тождественное 3×3": ([[0, 0, 0], [0, 1, 0], [0, 0, 0]], True, False, False),
    "Усреднение 3×3": ([[1] * 3 for _ in range(3)], True, False, False),
    "Усреднение 5×5": ([[1] * 5 for _ in range(5)], True, False, False),
    "Гаусс 3×3": ([[1, 2, 1], [2, 4, 2], [1, 2, 1]], True, False, False),
    "Гаусс 5×5": (_outer([1, 4, 6, 4, 1]), True, False, False),
    "Резкость": ([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], True, False, False),
    "Высокие частоты 3×3": ([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], True, False, True),
    "Лаплас (4 соседа)": ([[0, 1, 0], [1, -4, 1], [0, 1, 0]], True, False, True),
    "Лаплас (8 соседей)": ([[1, 1, 1], [1, -8, 1], [1, 1, 1]], True, False, True),
    "Собель X": (SOBEL_X, True, False, True),
    "Собель Y": (SOBEL_Y, True, False, True),
    "Тиснение": ([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], True, False, False),
}


def blur(arr, method, n, sigma=1.0):
    if method == "gauss":
        k = gauss_kernel_1d(n, sigma)
    else:
        k = box_kernel_1d(n)
    return convolve_separable(arr, k)


NEG_ADD128 = "+128"
NEG_ABS = "По модулю"
NEG_STRETCH = "Растяжение min..max"
NEG_CLIP = "Обрезка до 0"
NEG_MODES = [NEG_ADD128, NEG_ABS, NEG_STRETCH, NEG_CLIP]


def fix_negative(arr, mode):
    if mode == NEG_ADD128:
        return arr + 128.0
    if mode == NEG_ABS:
        return np.abs(arr)
    if mode == NEG_STRETCH:
        return stretch(arr)
    return arr


def high_pass(img, method, n, sigma, c, neg_mode):
    orig = img.astype(np.float32)
    blurred = blur(orig, method, n, sigma)
    hf = orig - c * blurred
    return to_uint8(blurred), to_uint8(fix_negative(hf, neg_mode))


def edges(img, pre_blur_sigma=0.0, threshold=0):
    g = grayscale(img)
    if pre_blur_sigma > 0:
        g = blur(g, "gauss", gauss_size(pre_blur_sigma), pre_blur_sigma)
    gx = convolve(g, SOBEL_X)
    gy = convolve(g, SOBEL_Y)
    mag = np.sqrt(gx * gx + gy * gy)
    mx = float(mag.max())
    if mx > 0:
        mag = mag * (255.0 / mx)
    binary = None
    if threshold > 0:
        binary = np.zeros(mag.shape, dtype=np.uint8)
        binary[mag >= threshold] = 255
    return to_uint8(mag), binary


def max_filter(arr, r):
    h, w = arr.shape[:2]
    pad = pad_edge(arr, (r, r), (0, 0))
    m = pad[0:h].copy()
    for i in range(1, 2 * r + 1):
        m = np.maximum(m, pad[i:i + h])
    pad = pad_edge(m, (0, 0), (r, r))
    out = pad[:, 0:w].copy()
    for j in range(1, 2 * r + 1):
        out = np.maximum(out, pad[:, j:j + w])
    return out


def harris(img, k=0.04, sigma=1.5, rel_threshold=0.01, nms_radius=4):
    g = grayscale(img)
    ix = convolve(g, SOBEL_X) / 8.0
    iy = convolve(g, SOBEL_Y) / 8.0
    win = gauss_kernel_1d(gauss_size(sigma), sigma)
    ixx = convolve_separable(ix * ix, win)
    iyy = convolve_separable(iy * iy, win)
    ixy = convolve_separable(ix * iy, win)
    trace = ixx + iyy
    r = ixx * iyy - ixy * ixy - k * trace * trace
    r_max = float(r.max())
    if r_max <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), r
    local_max = max_filter(r, nms_radius)
    mask = (r >= local_max) & (r > rel_threshold * r_max)
    ys, xs = np.nonzero(mask)
    return ys, xs, r


def response_to_image(r):
    pos = r.copy()
    pos[pos < 0] = 0
    mx = float(pos.max())
    if mx <= 0:
        return np.zeros(r.shape, dtype=np.uint8)
    return to_uint8(255.0 * (pos / mx) ** 0.25)


def draw_crosses(img, ys, xs, size=4, color=(255, 0, 0)):
    out = img.copy()
    if out.ndim == 2:
        out = np.stack([out, out, out], axis=-1)
    h, w = out.shape[:2]
    for d in range(-size, size + 1):
        cx = np.minimum(np.maximum(xs + d, 0), w - 1)
        cy = np.minimum(np.maximum(ys + d, 0), h - 1)
        out[ys, cx] = color
        out[cy, xs] = color
    return out
