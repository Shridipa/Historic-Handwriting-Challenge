import csv
from pathlib import Path

import cv2
import pandas as pd

from data_utils import DATA_ROOT, IMG_ROOT


def main():
    train = pd.read_csv(DATA_ROOT / 'data' / 'Train.csv', encoding='utf-8-sig', keep_default_na=False)
    rows = []
    for row in train.itertuples(index=False):
        path = IMG_ROOT / f'{row.ID}.jpg'
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        height, width = image.shape[:2]
        processed_width = 768
        timesteps = processed_width // 2
        target_length = len(str(row.Target))
        ratio = timesteps / max(target_length, 1)
        rows.append({'ID': row.ID, 'target_length': target_length, 'image_width': width, 'processed_width': processed_width, 'ctc_timesteps': timesteps, 'timestep_target_ratio': ratio, 'ctc_feasible': timesteps >= target_length, 'below_1_5x': ratio < 1.5})
    out = DATA_ROOT / 'runs' / 'ctc_capacity.csv'
    with out.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {len(rows)} rows to {out}')
    print(f'infeasible={sum(not row["ctc_feasible"] for row in rows)} below_1_5x={sum(row["below_1_5x"] for row in rows)} min_ratio={min(row["timestep_target_ratio"] for row in rows):.3f}')


if __name__ == '__main__':
    main()
