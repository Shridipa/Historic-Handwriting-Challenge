import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from data_utils import DATA_ROOT, IMG_ROOT, preprocess_image


def preserve_aspect(image, target_height=64, max_width=1536):
    height, width = image.shape[:2]
    new_width = max(1, round(width * target_height / max(height, 1)))
    scale = min(1.0, max_width / new_width)
    new_width = max(1, round(new_width * scale))
    resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
    return resized


def main():
    train = pd.read_csv(DATA_ROOT / 'data' / 'splits' / 'val.csv', encoding='utf-8', keep_default_na=False)
    train['_length'] = train['Target'].str.len()
    train['_width'] = train['ID'].map(lambda x: cv2.imread(str(IMG_ROOT / f'{x}.jpg'), cv2.IMREAD_GRAYSCALE).shape[1])
    train = train.sort_values(['_length', '_width'])
    positions = np.linspace(0, len(train) - 1, 20).round().astype(int)
    selected = train.iloc[positions]
    out_dir = DATA_ROOT / 'runs' / 'preprocessing_audit'
    examples = out_dir / 'examples'
    examples.mkdir(parents=True, exist_ok=True)
    rows = []
    for row_index, (_, row) in enumerate(selected.iterrows()):
        image = cv2.imread(str(IMG_ROOT / f"{row['ID']}.jpg"), cv2.IMREAD_GRAYSCALE)
        current = preprocess_image(image, target_height=64, target_width=768)
        adaptive = preserve_aspect(image)
        panels = [cv2.resize(image, (768, 64), interpolation=cv2.INTER_AREA), (current * 255).astype(np.uint8), adaptive]
        panels = [cv2.resize(panel, (768, 64), interpolation=cv2.INTER_AREA) for panel in panels]
        label = np.full((28, 2304), 255, dtype=np.uint8)
        cv2.putText(label, f"{row['ID']} len={len(row['Target'])}", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1, cv2.LINE_AA)
        rows.append(np.vstack([label, np.hstack(panels)]))
        cv2.imwrite(str(examples / f"{row['ID']}_original.png"), image)
        cv2.imwrite(str(examples / f"{row['ID']}_fixed.png"), current * 255)
        cv2.imwrite(str(examples / f"{row['ID']}_adaptive.png"), adaptive)
    cv2.imwrite(str(out_dir / 'comparison.png'), np.vstack(rows))
    print(f'wrote {len(selected)} examples to {out_dir}')


if __name__ == '__main__':
    main()
