import argparse
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from data_utils import DATA_ROOT, preprocess_image_aspect_preserving


def add_text(canvas, text, origin, width, color=(0, 0, 0), line_height=18, max_lines=4):
    x, y = origin
    lines = textwrap.wrap(str(text), width=width) or ['']
    for line in lines[:max_lines]:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        y += line_height


def make_panel(row, image_dir, output_dir, height, max_width):
    image = cv2.imread(str(image_dir / f'{row.ID}.jpg'), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_dir / f'{row.ID}.jpg')
    adaptive = preprocess_image_aspect_preserving(image, target_height=height, max_width=max_width)
    original_width = image.shape[1]
    original_height = image.shape[0]
    aspect = original_width / max(original_height, 1)
    processed_width = adaptive.shape[1]

    display_width = 1024
    original_display = cv2.resize(image, (display_width, max(1, round(image.shape[0] * display_width / image.shape[1]))), interpolation=cv2.INTER_AREA)
    adaptive_display = cv2.resize((adaptive * 255).astype(np.uint8), (display_width, height * display_width // max(processed_width, 1)), interpolation=cv2.INTER_NEAREST)
    image_height = max(original_display.shape[0], adaptive_display.shape[0])
    original_display = cv2.copyMakeBorder(original_display, 0, image_height - original_display.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=255)
    adaptive_display = cv2.copyMakeBorder(adaptive_display, 0, image_height - adaptive_display.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=255)
    images = np.hstack([cv2.cvtColor(original_display, cv2.COLOR_GRAY2BGR), cv2.cvtColor(adaptive_display, cv2.COLOR_GRAY2BGR)])

    text_height = 145
    label = np.full((text_height, images.shape[1], 3), 255, dtype=np.uint8)
    add_text(label, f'ID={row.ID}  CER={row.CER:.4f}  ratio={row.timestep_target_ratio:.2f}  aspect={aspect:.2f}', (8, 20), 130)
    add_text(label, f'original={original_width}x{original_height}  adaptive={processed_width}x{height}', (8, 40), 130)
    add_text(label, f'Target: {row.Target}', (8, 64), 145, max_lines=3)
    add_text(label, f'Prediction: {row.Prediction}', (8, 118), 145, max_lines=2)
    panel = np.vstack([label, images])
    output_path = output_dir / f'{row.ID}.png'
    cv2.imwrite(str(output_path), panel)
    return panel


def select_cohorts(audit, count):
    audit = audit.copy()
    audit['aspect_ratio'] = audit['original_width'] / audit['original_height'].clip(lower=1)
    selected = {}
    selected['worst_cer'] = audit.sort_values('CER', ascending=False).head(count)
    low_ratio = audit.sort_values('timestep_target_ratio')
    selected['low_ratio'] = low_ratio.head(count)
    compact = audit[audit['aspect_ratio'] < 12].sort_values('CER', ascending=False)
    selected['compact'] = compact.head(count)

    audit['length_group'] = pd.cut(audit['target_length'], [-1, 59, 89, float('inf')], labels=['short', 'medium', 'long'])
    audit['aspect_group'] = pd.cut(audit['aspect_ratio'], [-1, 12, 20, float('inf')], labels=['compact', 'middle', 'wide'])
    good = audit.sort_values('CER')
    selected['matched_good'] = good.groupby(['length_group', 'aspect_group'], observed=False, group_keys=False).head(max(1, count // 5)).head(count)

    frames = []
    for cohort, rows in selected.items():
        rows = rows.copy()
        rows['cohort'] = cohort
        frames.append(rows)
    manifest = pd.concat(frames, ignore_index=True).drop_duplicates('ID')
    return selected, manifest


def main():
    parser = argparse.ArgumentParser(description='Render focused visual audit cohorts for adaptive OCR.')
    parser.add_argument('--run', required=True)
    parser.add_argument('--image-dir', default=str(DATA_ROOT / 'data' / 'images'))
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--max-width', type=int, default=1024)
    parser.add_argument('--count', type=int, default=20)
    args = parser.parse_args()

    run = Path(args.run)
    audit = pd.read_csv(run / 'ctc_capacity_validation.csv', keep_default_na=False)
    selected, manifest = select_cohorts(audit, args.count)
    out_dir = run / 'visual_audit'
    panels_dir = out_dir / 'panels'
    panels_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / 'manifest.csv', index=False, encoding='utf-8')

    for cohort, rows in selected.items():
        cohort_dir = panels_dir / cohort
        cohort_dir.mkdir(parents=True, exist_ok=True)
        cohort_panels = []
        for row in rows.itertuples(index=False):
            panel = make_panel(row, Path(args.image_dir), cohort_dir, args.height, args.max_width)
            cohort_panels.append(panel)
        if cohort_panels:
            width = max(panel.shape[1] for panel in cohort_panels)
            normalized = [cv2.copyMakeBorder(panel, 0, 0, 0, width - panel.shape[1], cv2.BORDER_CONSTANT, value=255) for panel in cohort_panels]
            cv2.imwrite(str(out_dir / f'{cohort}.png'), np.vstack(normalized))
        print(f'{cohort}: {len(rows)} panels')
    print(f'wrote visual audit to {out_dir}')


if __name__ == '__main__':
    main()