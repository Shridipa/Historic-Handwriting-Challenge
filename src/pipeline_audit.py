import json
import unicodedata
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_utils import DATA_ROOT, IMG_ROOT, build_charset


def audit():
    train_path = DATA_ROOT / 'data' / 'Train.csv'
    test_path = DATA_ROOT / 'data' / 'Test.csv'
    train = pd.read_csv(train_path, encoding='utf-8-sig', keep_default_na=False)
    test = pd.read_csv(test_path, encoding='utf-8-sig', keep_default_na=False)
    lengths = train['Target'].astype(str).str.len()
    dimensions = []
    missing = []
    undecodable = []
    for sample_id in train['ID'].astype(str):
        path = IMG_ROOT / f'{sample_id}.jpg'
        if not path.exists():
            missing.append(sample_id)
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            undecodable.append(sample_id)
        else:
            height, width = image.shape[:2]
            dimensions.append((sample_id, width, height, width / max(height, 1)))

    texts = train['Target'].astype(str)
    chars = Counter(''.join(texts))
    normalized_changes = Counter()
    controls = Counter()
    for text in texts:
        for char in text:
            normalized = unicodedata.normalize('NFC', char)
            if normalized != char:
                normalized_changes[f'{char!r} -> {normalized!r}'] += 1
            if unicodedata.category(char).startswith('C'):
                controls[repr(char)] += 1

    train_ids, val_ids = train_test_split(train['ID'].astype(str), test_size=0.1, random_state=42, shuffle=True)
    val_set = set(val_ids)
    train_chars = set(''.join(train.loc[train['ID'].astype(str).isin(set(train_ids)), 'Target'].astype(str)))
    val_chars = set(''.join(train.loc[train['ID'].astype(str).isin(val_set), 'Target'].astype(str)))
    widths = [item[1] for item in dimensions]
    heights = [item[2] for item in dimensions]
    ratios = [item[3] for item in dimensions]
    report = {
        'train_rows': len(train),
        'test_rows': len(test),
        'missing_images': len(missing),
        'undecodable_images': len(undecodable),
        'duplicate_image_ids': int(train['ID'].duplicated().sum()),
        'duplicate_transcriptions': int(train['Target'].duplicated().sum()),
        'unique_transcriptions': int(train['Target'].nunique()),
        'empty_transcriptions': int((texts.str.strip() == '').sum()),
        'target_length': {'min': int(lengths.min()), 'max': int(lengths.max()), 'mean': float(lengths.mean()), 'median': float(lengths.median())},
        'image_dimensions': {'min_width': int(min(widths)), 'max_width': int(max(widths)), 'mean_width': float(np.mean(widths)), 'min_height': int(min(heights)), 'max_height': int(max(heights)), 'mean_height': float(np.mean(heights))},
        'aspect_ratio': {'min': float(min(ratios)), 'max': float(max(ratios)), 'mean': float(np.mean(ratios)), 'median': float(np.median(ratios))},
        'characters': [{'char': char, 'codepoint': f'U+{ord(char):04X}', 'frequency': count} for char, count in sorted(chars.items())],
        'validation_only_characters': sorted(val_chars - train_chars),
        'unicode_normalization_changes': dict(normalized_changes),
        'control_characters': dict(controls),
        'split': {'train_rows': len(train_ids), 'val_rows': len(val_ids), 'random_state': 42},
    }
    out = DATA_ROOT / 'runs' / 'pipeline_audit.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    split_dir = DATA_ROOT / 'data' / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    train[train['ID'].astype(str).isin(set(train_ids))].to_csv(split_dir / 'train.csv', index=False, encoding='utf-8')
    train[train['ID'].astype(str).isin(val_set)].to_csv(split_dir / 'val.csv', index=False, encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    audit()
