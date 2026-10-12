import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm


DATA_ROOT = Path(__file__).resolve().parents[1]
IMG_ROOT = DATA_ROOT / 'images' / 'images'
TRAIN_CSV = DATA_ROOT / 'road-barbados-historic-handwriting-challenge20260703-3588-9x1pqe' / 'Train.csv'
TEST_CSV = DATA_ROOT / 'road-barbados-historic-handwriting-challenge20260703-3588-9x1pqe' / 'Test.csv'


def list_images():
    return sorted(IMG_ROOT.glob('*.jpg'))


def load_split_df(path):
    return pd.read_csv(path, encoding='utf-8-sig')


def ensure_expected_layout():
    DATA_ROOT.joinpath('data').mkdir(exist_ok=True)
    if not DATA_ROOT.joinpath('data', 'Train.csv').exists():
        import shutil
        src = TRAIN_CSV
        dst = DATA_ROOT / 'data' / 'Train.csv'
        shutil.copy2(src, dst)
    if not DATA_ROOT.joinpath('data', 'Test.csv').exists():
        import shutil
        src = TEST_CSV
        dst = DATA_ROOT / 'data' / 'Test.csv'
        shutil.copy2(src, dst)
    if not DATA_ROOT.joinpath('data', 'SampleSubmission.csv').exists():
        import shutil
        src = DATA_ROOT / 'road-barbados-historic-handwriting-challenge20260703-3588-9x1pqe' / 'SampleSubmission.csv'
        dst = DATA_ROOT / 'data' / 'SampleSubmission.csv'
        shutil.copy2(src, dst)
    data_img_dir = DATA_ROOT / 'data' / 'images'
    data_img_dir.mkdir(exist_ok=True)
    if not any(data_img_dir.iterdir()):
        for img in list_images():
            target = data_img_dir / img.name
            if not target.exists():
                import shutil
                shutil.copy2(img, target)


def build_charset(train_df):
    chars = sorted({ch for text in train_df['Target'].fillna('').astype(str) for ch in text})
    return ''.join(chars)


def sample_image(id_value, size=None):
    path = IMG_ROOT / f'{id_value}.jpg'
    if not path.exists():
        raise FileNotFoundError(path)
    img = Image.open(path).convert('L')
    if size is not None:
        img = img.resize(size, Image.Resampling.BILINEAR)
    return np.array(img)


def preprocess_image(image: np.ndarray, target_height=64, target_width=768):
    img = image.astype(np.float32)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.max() > 1.0:
        img = img / 255.0
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    width = target_width if target_width is not None else max(1, int(img.shape[1] * (target_height / max(img.shape[0], 1))))
    img = cv2.resize(img, (width, target_height), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def preprocess_image_aspect_preserving(image: np.ndarray, target_height=64, max_width=1536):
    img = image.astype(np.float32)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.max() > 1.0:
        img = img / 255.0
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    width = max(1, round(img.shape[1] * target_height / max(img.shape[0], 1)))
    width = min(width, max_width)
    img = cv2.resize(img, (width, target_height), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def _unsharp_mask(img: np.ndarray, amount: float = 1.2, sigma: float = 1.0) -> np.ndarray:
    """img: float32 HxW in [0,1]. Returns sharpened copy (used by Phase 6 variant)."""
    k = 2 * int(2 * sigma) + 1
    blur = cv2.GaussianBlur(img, (k, k), sigma)
    return np.clip(img + amount * (img - blur), 0.0, 1.0).astype(np.float32)


def preprocess_image_aspect_preserving_bgcontrast_sharp(image: np.ndarray, target_height=64, max_width=1536):
    """bgcontrast pipeline + unsharp mask AFTER the final resize (Phase 6 variant).

    Motivation (error analysis): blurred originals show CER 0.18 vs 0.087 for sharp
    ones; Spearman(sharpness, CER) = -0.40. Unsharp masking at the final resolution
    keeps the sharpening scale consistent across target heights.
    """
    img = preprocess_image_aspect_preserving_bgcontrast(image, target_height=target_height, max_width=max_width)
    return _unsharp_mask(img, amount=1.2, sigma=1.0)


def preprocess_image_aspect_preserving_bgcontrast(image: np.ndarray, target_height=64, max_width=1536):
    img = image.astype(np.float32)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.max() > 1.0:
        img = img / 255.0
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply((img * 255.0).astype(np.uint8)).astype(np.float32) / 255.0
    width = max(1, round(img.shape[1] * target_height / max(img.shape[0], 1)))
    width = min(width, max_width)
    img = cv2.resize(img, (width, target_height), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def make_char_to_index(charset):
    chars = sorted(set(charset))
    return {ch: i for i, ch in enumerate(chars)}


def make_index_to_char(charset):
    return {i: ch for i, ch in enumerate(sorted(set(charset)))}
