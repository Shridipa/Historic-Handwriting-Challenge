import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd

from metrics import _levenshtein_distance, cer, wer


def align_chars(reference, hypothesis):
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i, ref_char in enumerate(reference, 1):
        for j, hyp_char in enumerate(hypothesis, 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (ref_char != hyp_char),
            )
    pairs = []
    i, j = len(reference), len(hypothesis)
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            pairs.append((reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((reference[i - 1], '<DEL>'))
            i -= 1
        else:
            pairs.append(('<INS>', hypothesis[j - 1]))
            j -= 1
    return reversed(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--train-csv', default='data/Train.csv')
    parser.add_argument('--image-dir', default='data/images')
    args = parser.parse_args()
    run = Path(args.run)
    predictions = pd.read_csv(run / 'val_predictions.csv', keep_default_na=False)
    train = pd.read_csv(args.train_csv, encoding='utf-8-sig')
    image_dir = Path(args.image_dir)

    errors = []
    confusion = Counter()
    stratified = []
    for row in predictions.itertuples(index=False):
        target = str(row.Target)
        prediction = str(row.Prediction)
        image = cv2.imread(str(image_dir / f'{row.ID}.jpg'), cv2.IMREAD_GRAYSCALE)
        height, width = image.shape[:2]
        ratio = width / max(height, 1)
        errors.append({'ID': row.ID, 'Prediction': prediction, 'Target': target, 'CER': cer(target, prediction), 'WER': wer(target, prediction)})
        confusion.update((a, b) for a, b in align_chars(target, prediction) if a != b)
        stratified.append({'length_bucket': '37-59' if len(target) < 60 else '60-89' if len(target) < 90 else '90-120', 'aspect_bucket': '<12' if ratio < 12 else '12-20' if ratio < 20 else '>=20', 'CER': cer(target, prediction), 'count': 1})

    errors_df = pd.DataFrame(errors).sort_values(['CER', 'WER'], ascending=False)
    errors_df.to_csv(run / 'validation_errors.csv', index=False, encoding='utf-8')
    pd.DataFrame([{'reference': a, 'hypothesis': b, 'count': n} for (a, b), n in confusion.most_common(20)]).to_csv(run / 'confusion_top20.csv', index=False)
    stratified_df = pd.DataFrame(stratified).groupby(['length_bucket', 'aspect_bucket'], as_index=False).agg(CER=('CER', 'mean'), count=('count', 'sum'))
    stratified_df.to_csv(run / 'stratification.csv', index=False)

    print('worst_30:')
    print(errors_df.head(30)[['ID', 'CER', 'WER', 'Prediction', 'Target']].to_string(index=False))
    print('\nconfusion_top20:')
    print(pd.read_csv(run / 'confusion_top20.csv').to_string(index=False))
    print('\nstratification:')
    print(stratified_df.to_string(index=False))


if __name__ == '__main__':
    main()
