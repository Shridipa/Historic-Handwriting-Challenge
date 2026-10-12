"""Phase 4 error-category and image-quality analysis.

Consumes a val_predictions CSV (ID, Target, Prediction) plus the raw images and produces:
  - per-example edit-operation counts and error-category flags
  - top character confusions
  - error-category totals
  - image-quality metrics (brightness, contrast, sharpness, ink ratio, size, aspect)
  - Spearman correlation of CER with quality metrics
  - CER by quality and length buckets
Outputs are written to an output directory as CSV + a readable summary markdown.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from metrics import cer, wer
from diagnose_run import align_chars


def edit_ops(reference: str, hypothesis: str):
    subs = ins = dels = 0
    for a, b in align_chars(reference, hypothesis):
        if a == b:
            continue
        if a == '<INS>':
            ins += 1
        elif b == '<DEL>':
            dels += 1
        else:
            subs += 1
    return subs, ins, dels


def error_categories(reference: str, hypothesis: str):
    subs, ins, dels = edit_ops(reference, hypothesis)
    pairs = [(a, b) for a, b in align_chars(reference, hypothesis) if a != b]
    case_pairs = sum(1 for a, b in pairs if a != '<INS>' and b != '<DEL>' and a.lower() == b.lower())
    space_pairs = sum(1 for a, b in pairs if (a == ' ') != (b == ' '))
    punct_pairs = sum(1 for a, b in pairs
                      if a != '<INS>' and b != '<DEL>'
                      and not a.isalnum() and not b.isalnum() and a != ' ' and b != ' ')
    # repeated-character errors: same char expected where model emitted duplicate streak
    repeat_pairs = 0
    ref_chars = [a for a, b in [(x, y) for x, y in align_chars(reference, hypothesis)] ]
    for i, (a, b) in enumerate(pairs):
        if a != '<INS>' and b != '<DEL>':
            # detect near-duplicate confusion like 'nn'->'n' handled as del; approximate via doubled reference
            pass
    doubled_ref = sum(1 for i in range(1, len(reference)) if reference[i] == reference[i - 1])
    return {
        'subs': subs,
        'insertions': ins,
        'deletions': dels,
        'case_errors': case_pairs,
        'space_errors': space_pairs,
        'punctuation_errors': punct_pairs,
        'doubled_char_positions': doubled_ref,
    }


def image_quality(image_path: Path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    brightness = float(img.mean())
    contrast = float(img.std())
    sharpness = float(cv2.Laplacian(img, cv2.CV_64F).var())
    otsu_t, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio = float(binary.mean() / 255.0)
    return {
        'img_width': w,
        'img_height': h,
        'aspect': w / max(h, 1),
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': sharpness,
        'ink_ratio': ink_ratio,
    }


def quality_bucket(value, edges, labels):
    for i, edge in enumerate(edges):
        if value < edge:
            return labels[i]
    return labels[-1]


def analyze(pred_csv: Path, image_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(pred_csv, keep_default_na=False)
    rows = []
    confusion = Counter()
    cat_totals = Counter()
    for row in df.itertuples(index=False):
        target = str(row.Target)
        pred = str(row.Prediction)
        q = image_quality(image_dir / f'{row.ID}.jpg') or {}
        cats = error_categories(target, pred)
        for a, b in align_chars(target, pred):
            if a != b:
                confusion[(a, b)] += 1
        cat_totals.update({k: v for k, v in cats.items()})
        rows.append({
            'ID': row.ID,
            'CER': cer(target, pred),
            'WER': wer(target, pred),
            'target_len': len(target),
            'edit_distance': cats['subs'] + cats['insertions'] + cats['deletions'],
            **cats,
            **q,
        })
    stats = pd.DataFrame(rows)
    stats.to_csv(out_dir / 'error_quality_stats.csv', index=False, encoding='utf-8')

    conf_df = pd.DataFrame(
        [{'reference': a, 'hypothesis': b, 'count': n} for (a, b), n in confusion.most_common(30)])
    conf_df.to_csv(out_dir / 'confusion_top30.csv', index=False, encoding='utf-8')

    cat_df = pd.DataFrame([cat_totals]).T.reset_index()
    cat_df.columns = ['category', 'count']
    cat_df.to_csv(out_dir / 'error_categories.csv', index=False, encoding='utf-8')

    corr = stats[['CER', 'target_len', 'img_width', 'img_height', 'aspect',
                  'brightness', 'contrast', 'sharpness', 'ink_ratio']].corr(method='spearman')['CER']
    corr.drop('CER').to_csv(out_dir / 'quality_correlations.csv', header=['spearman_vs_cer'], encoding='utf-8')

    # Buckets
    stats['len_bucket'] = pd.cut(stats['target_len'], [0, 49, 69, 200],
                                 labels=['short(<50)', 'medium(50-69)', 'long(>=70)'])
    stats['contrast_bucket'] = [quality_bucket(v, [40], ['low_contrast', 'high_contrast']) for v in stats['contrast']]
    stats['sharp_bucket'] = [quality_bucket(v, [100, 500], ['blurred', 'medium', 'sharp']) for v in stats['sharpness']]
    stats['width_bucket'] = pd.cut(stats['img_width'], [0, 1000, 1500, 3000, 10000],
                                   labels=['<1000', '1000-1500', '1500-3000', '>=3000'])
    bucket_cols = {}
    for col in ['len_bucket', 'contrast_bucket', 'sharp_bucket', 'width_bucket']:
        g = stats.groupby(col, observed=True)['CER'].agg(['mean', 'count'])
        g.to_csv(out_dir / f'cer_by_{col}.csv', encoding='utf-8')
        bucket_cols[col] = g

    summary = {
        'n_examples': int(len(stats)),
        'corpus_cer': float(cer(' '.join(df['Target'].astype(str)), ' '.join(df['Prediction'].astype(str))) if False else stats['edit_distance'].sum() / max(stats['target_len'].sum(), 1)),
        'mean_example_cer': float(stats['CER'].mean()),
        'categories': dict(cat_totals),
        'spearman_cer_corr': {k: round(float(v), 4) for k, v in corr.drop('CER').items()},
        'top_confusions': [(a, b, int(n)) for (a, b), n in confusion.most_common(10)],
        'buckets': {col: {str(k): {'cer': round(float(v['mean']), 4), 'n': int(v['count'])}
                          for k, v in g.iterrows()} for col, g in bucket_cols.items()},
    }
    with open(out_dir / 'error_analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='val_predictions CSV with ID, Target, Prediction')
    parser.add_argument('--image-dir', default='data/images')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    analyze(Path(args.pred), Path(args.image_dir), Path(args.out))


if __name__ == '__main__':
    main()
