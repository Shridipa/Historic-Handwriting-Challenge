"""Training-time image augmentations (Phase 7). CPU-only.

Applied AFTER preprocessing (on the processed float32 HxW image in [0,1]),
BEFORE tensor conversion. Only ever applied to the TRAINING dataset.

Ops (enable via --augment a,b,c in train_crnn.py):
  rotate    small in-plane rotation, +/-1 deg
  shear     slight vertical shear, +/-0.02
  stretch   horizontal rescale of the processed line, 0.92-1.08x
  penweight morphological thin/thick (simulates pen pressure/bleed)
  blur      extra gaussian blur sigma ~ U(0.5, 1.2)
  noise     additive gaussian noise, sigma ~ U(0.008, 0.03)
  contrast  brightness/contrast jitter
"""
import cv2
import numpy as np

OPS = ('rotate', 'shear', 'stretch', 'penweight', 'blur', 'noise', 'contrast')


def augment_image(img: np.ndarray, ops, rng: np.random.Generator) -> np.ndarray:
    """img: float32 HxW in [0,1] (dark ink on light paper after preprocessing)."""
    img = img.copy()
    h, w = img.shape
    if 'rotate' in ops and rng.random() < 0.5:
        ang = float(rng.uniform(-1.0, 1.0))
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=1.0)
    if 'shear' in ops and rng.random() < 0.3:
        sh = float(rng.uniform(-0.02, 0.02))
        M = np.float32([[1.0, sh, -sh * h / 2.0], [0.0, 1.0, 0.0]])
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=1.0)
    if 'stretch' in ops and rng.random() < 0.5:
        new_w = max(8, int(round(w * float(rng.uniform(0.92, 1.08)))))
        img = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA if new_w < w else cv2.INTER_LINEAR)
    if 'penweight' in ops:
        r = rng.random()
        if r < 0.25:      # thinner strokes (dilate bright background)
            img = cv2.dilate(img, np.ones((3, 3), np.uint8))
        elif r < 0.5:     # thicker strokes (erode bright background)
            img = cv2.erode(img, np.ones((3, 3), np.uint8))
    if 'blur' in ops and rng.random() < 0.3:
        sigma = float(rng.uniform(0.5, 1.2))
        k = 2 * int(2 * sigma) + 1
        img = cv2.GaussianBlur(img, (k, k), sigma)
    if 'noise' in ops and rng.random() < 0.4:
        img = img + rng.normal(0.0, float(rng.uniform(0.008, 0.03)), img.shape).astype(np.float32)
    if 'contrast' in ops and rng.random() < 0.5:
        alpha = float(rng.uniform(0.9, 1.1))
        beta = float(rng.uniform(-0.05, 0.05))
        img = alpha * img + beta
    return np.clip(img, 0.0, 1.0).astype(np.float32)
