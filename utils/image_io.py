import numpy as np
from PIL import Image

def load_image(path):
    return np.array(Image.open(path).convert("RGB"))

def rgb_to_grayscale(img):
    """Конвертация RGB в ч/б по формуле яркости"""
    # Формула: Y = 0.299*R + 0.587*G + 0.114*B
    return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)