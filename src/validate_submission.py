import argparse
import pandas as pd
from pathlib import Path


def validate_submission(submission_path='submission.csv', test_csv='data/Test.csv'):
    sub = pd.read_csv(submission_path, keep_default_na=False)
    test = pd.read_csv(test_csv, encoding='utf-8-sig', keep_default_na=False)

    sub['Target'] = sub['Target'].fillna('')
    test['ID'] = test['ID'].fillna('')

    required_cols = ['ID', 'Target']
    ok = True
    if list(sub.columns) != required_cols:
        print('FAIL: columns mismatch')
        ok = False
    if len(sub) != len(test):
        print('FAIL: row count mismatch')
        ok = False
    if set(sub['ID']) != set(test['ID']):
        print('FAIL: ID mismatch')
        ok = False
    if sub['ID'].duplicated().any():
        print('FAIL: duplicated IDs')
        ok = False
    if sub.isna().any().any():
        print('FAIL: NaN values present')
        ok = False

    if ok:
        print('PASS')
        return 0
    print('FAIL')
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', default='submission.csv')
    parser.add_argument('--test-csv', default='data/Test.csv')
    args = parser.parse_args()
    raise SystemExit(validate_submission(args.submission, args.test_csv))
