import argparse
from pathlib import Path

import cv2
import pandas as pd

from data_utils import DATA_ROOT
from metrics import cer


def capacity_row(row, image_dir, height, max_width):
    image = cv2.imread(str(image_dir / f'{row.ID}.jpg'), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_dir / f'{row.ID}.jpg')
    original_height, original_width = image.shape[:2]
    processed_width = max(1, min(max_width, round(original_width * height / max(original_height, 1))))
    target = str(row.Target)
    timesteps = processed_width // 2
    ratio = timesteps / max(len(target), 1)
    return {
        'ID': row.ID,
        'target_length': len(target),
        'original_width': original_width,
        'original_height': original_height,
        'processed_width': processed_width,
        'ctc_timesteps': timesteps,
        'timestep_target_ratio': ratio,
        'ctc_feasible': timesteps >= len(target),
    }


def main():
    parser = argparse.ArgumentParser(description='Join validation CER with adaptive CTC capacity.')
    parser.add_argument('--run', required=True, help='Run directory containing val_predictions.csv')
    parser.add_argument('--image-dir', default=str(DATA_ROOT / 'data' / 'images'))
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--max-width', type=int, default=1024)
    args = parser.parse_args()

    run = Path(args.run)
    predictions = pd.read_csv(run / 'val_predictions.csv', keep_default_na=False)
    rows = []
    for row in predictions.itertuples(index=False):
        result = capacity_row(row, Path(args.image_dir), args.height, args.max_width)
        result['CER'] = cer(str(row.Target), str(row.Prediction))
        result['Prediction'] = row.Prediction
        result['Target'] = row.Target
        rows.append(result)

    audit = pd.DataFrame(rows)
    audit['capacity_bin'] = pd.cut(
        audit['timestep_target_ratio'],
        bins=[-float('inf'), 4.0, 5.0, 6.0, 8.0, float('inf')],
        labels=['<4', '4-5', '5-6', '6-8', '>=8'],
        right=False,
    )
    output = run / 'ctc_capacity_validation.csv'
    audit.to_csv(output, index=False, encoding='utf-8')

    summary = audit.groupby('capacity_bin', observed=False).agg(
        CER=('CER', 'mean'),
        samples=('ID', 'size'),
        mean_ratio=('timestep_target_ratio', 'mean'),
        min_ratio=('timestep_target_ratio', 'min'),
        max_ratio=('timestep_target_ratio', 'max'),
        infeasible=('ctc_feasible', lambda values: int((~values).sum())),
    ).reset_index()

    print(f'wrote {len(audit)} rows to {output}')
    print(f"infeasible={int((~audit['ctc_feasible']).sum())} below_1.5x={int((audit['timestep_target_ratio'] < 1.5).sum())}")
    print(f"ratio_cer_pearson={audit['timestep_target_ratio'].corr(audit['CER']):.4f}")
    print('\nCER by timestep/target-length ratio:')
    print(summary.to_string(index=False))
    print('\nWorst examples:')
    print(audit.sort_values(['CER', 'timestep_target_ratio'], ascending=[False, True]).head(20)[
        ['ID', 'CER', 'timestep_target_ratio', 'ctc_timesteps', 'target_length', 'Prediction', 'Target']
    ].to_string(index=False))


if __name__ == '__main__':
    main()