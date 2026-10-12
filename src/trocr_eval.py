"""Phase 9: pretrained TrOCR baseline on validation (or train subset).

The pretrained model EARNs adoption only by beating the CRNN on validation Final.
Metrics identical to the CRNN pipeline (src/metrics.py compute_corpus_metrics).
Logits/text are evaluated as-is (no charset projection) for the measurement;
charset projection happens later only if adopted (Phase 12+ ensemble).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import compute_corpus_metrics  # noqa: E402

MODEL_ID = 'microsoft/trocr-base-handwritten'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=str(Path('data') / 'splits' / 'val.csv'))
    parser.add_argument('--image-dir', default=str(Path('data') / 'images'))
    parser.add_argument('--limit', type=int, default=0, help='0 = all rows')
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--out', required=True, help='output dir, e.g. runs/pretrained_trocr')
    args = parser.parse_args()

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, encoding='utf-8-sig')
    if args.limit:
        df = df.head(args.limit)
    ids = df['ID'].astype(str).tolist()
    refs = df['Target'].fillna('').astype(str).tolist()

    processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    model.eval()
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))

    hyps = []
    t0 = time.time()
    for i, rid in enumerate(tqdm(ids, desc='trocr')):
        img_bgr = cv2.imread(str(Path(args.image_dir) / f'{rid}.jpg'))
        img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        pixel_values = processor(images=img, return_tensors='pt').pixel_values
        with torch.no_grad():
            generated = model.generate(pixel_values, max_length=args.max_length)
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        hyps.append(text)
        # incremental partial-save every 50 rows so interruptions keep partial results
        if (i + 1) % 50 == 0:
            cer, wer, final = compute_corpus_metrics(refs[:i + 1], hyps[:i + 1])
            with open(out_dir / 'partial_run.json', 'w', encoding='utf-8') as f:
                json.dump({'rows': i + 1, 'cer': cer, 'wer': wer, 'final': final,
                           'elapsed_s': time.time() - t0}, f, indent=2)

    elapsed = time.time() - t0
    cer, wer, final = compute_corpus_metrics(refs, hyps)
    print(f'TrOCR val: CER={cer:.4f} WER={wer:.4f} Final={final:.4f} '
          f'({elapsed:.0f} s, {elapsed / len(ids):.2f} s/img, {len(ids)} rows)')

    result = {'model': MODEL_ID, 'split_csv': args.csv, 'rows': len(ids),
              'cer': cer, 'wer': wer, 'final': final,
              'seconds': elapsed, 's_per_img': elapsed / len(ids)}
    with open(out_dir / 'val_eval.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    with open(out_dir / 'predictions.csv', 'w', encoding='utf-8', newline='') as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(['ID', 'Target', 'Prediction'])
        w.writerows(zip(ids, refs, hyps))


if __name__ == '__main__':
    main()
