"""CTC prefix beam-search decoding (Phase 10) + LM fusion (Phase 11).

Computes model logits once per image (cached to npz), then decodes the cached
log-softmax with several beam widths and reports validation CER/WER/Final per
width. LM fusion uses an n-gram model trained ONLY on data/splits/train.csv
transcripts; lambda is tuned on validation.

Usage:
  # benchmark beam widths on validation
  python src/beam_search.py --checkpoint runs/adaptive_1024_bgcontrast_h80/checkpoints/best.pt \
      --beams 1,5,10,25,50 --out runs/beam_search/baseline

  # lambda sweep for LM fusion (validation)
  python src/beam_search.py --checkpoint ... --beams 10 --lm-path runs/lm/fusion_lm.pkl \
      --lm-weights 0,0.05,0.1,0.2,0.3,0.5 --out runs/beam_search/lm_sweep

  # generate a test-split submission with beam+LM decoding
  python src/beam_search.py --checkpoint ... --mode test --beam 10 \
      --lm-path runs/lm/fusion_lm.pkl --lm-weight 0.2 --submission-out submission_beam_lm.csv \
      --out runs/beam_search/test_out
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from infer_crnn import load_model, preprocess
from lm_ngram import FusionLM
from metrics import compute_corpus_metrics


NEG_INF = float('-inf')


def decode_logits(log_probs: np.ndarray, charset, beam_width: int, top_classes: int = 12,
                  lm: 'FusionLM | None' = None, lm_weight: float = 0.0) -> str:
    """log_probs: [T, C] log-softmax output.

    top_classes restricts candidate characters per timestep to the highest-probability
    classes (plus blank). Beyond the top few classes log-probs are far below the beam
    cutoff, so results are unchanged in practice while CPU time drops by ~an order of
    magnitude (this benchmark runs concurrently with training).
    """
    idx_to_char = {i + 1: ch for i, ch in enumerate(charset)}
    if beam_width <= 1:
        seq = log_probs.argmax(axis=-1)
        out = []
        prev = None
        for t in seq:
            if t == 0:
                prev = None
                continue
            if t == prev:
                continue
            out.append(idx_to_char.get(int(t), ''))
            prev = int(t)
        return ''.join(out)

    beams = {(): (0.0, NEG_INF, '')}  # prefix -> (logp_blank, logp_nonblank, decoded_text)
    T, C = log_probs.shape
    cand_all = np.argsort(log_probs, axis=-1)[:, -top_classes:]
    use_lm = lm is not None and lm_weight != 0.0
    for t in range(T):
        lp = log_probs[t]
        cand = [int(c) for c in cand_all[t]]
        next_beams = defaultdict(lambda: (NEG_INF, NEG_INF, None))
        for prefix, (pb, pnb, text) in beams.items():
            # blank: both states collapse into the blank state of the same prefix
            nb, nnb, _ = next_beams[prefix]
            next_beams[prefix] = (np.logaddexp(nb, np.logaddexp(pb, pnb) + lp[0]), nnb, text)
            last = prefix[-1] if prefix else None
            for c in cand:
                if c == 0:
                    continue
                p = lp[c]
                ch = idx_to_char.get(c, '')
                if c == last:
                    # repeat: non-blank stays in prefix (no char emitted, no LM term);
                    # blank-state mass extends to new prefix (emits ch -> LM applies)
                    nb, nnb, txt = next_beams[prefix]
                    next_beams[prefix] = (nb, np.logaddexp(nnb, pnb + p), txt)
                    lm_b = lm_weight * lm.score_transition(text, ch) if use_lm else 0.0
                    new_prefix = prefix + (c,)
                    nb2, nnb2, txt2 = next_beams[new_prefix]
                    next_beams[new_prefix] = (nb2, np.logaddexp(nnb2, pb + p + lm_b), text + ch)
                else:
                    lm_b = lm_weight * lm.score_transition(text, ch) if use_lm else 0.0
                    new_prefix = prefix + (c,)
                    nb2, nnb2, txt2 = next_beams[new_prefix]
                    next_beams[new_prefix] = (nb2, np.logaddexp(nnb2, np.logaddexp(pb, pnb) + p + lm_b), text + ch)
        top = sorted(next_beams.items(),
                     key=lambda kv: max(kv[1][0], kv[1][1]), reverse=True)[:beam_width]
        beams = dict(top)
    if use_lm:
        # add word-LM EOS / trailing-word score for final ranking
        best = max(beams.items(),
                   key=lambda kv: max(kv[1][0], kv[1][1]) + lm_weight * lm.final_score(kv[1][2]))[0]
    else:
        best = max(beams.items(), key=lambda kv: max(kv[1][0], kv[1][1]))[0]
    return ''.join(idx_to_char.get(c, '') for c in best)


def get_log_probs(df, model, config, args, out_dir):
    """Forward pass over df rows with an npz cache keyed on checkpoint+row IDs."""
    ids = df['ID'].astype(str).tolist()
    ckpt_mtime = Path(args.checkpoint).stat().st_mtime
    cache_meta = {
        'checkpoint': str(args.checkpoint),
        'checkpoint_mtime': ckpt_mtime,
        'val_csv': str(args.val_csv),
        'ids': ids,
    }
    # Filename keyed on row IDs (checkpoint mtime lives in the metadata), so a
    # --limit subset can't clobber the full-split cache.
    ids_key = hashlib.md5('|'.join(ids).encode('utf-8')).hexdigest()[:8]
    cache_path = None if args.no_cache else out_dir / f'log_probs_cache_{len(ids)}_{ids_key}.npz'

    all_log_probs = None
    if cache_path is not None and cache_path.exists():
        try:
            data = np.load(cache_path)
            meta = json.loads(str(data['meta'][0]))
            if (meta.get('ids') == cache_meta['ids']
                    and meta.get('checkpoint') == cache_meta['checkpoint']
                    and meta.get('val_csv') == cache_meta['val_csv']
                    and abs(meta.get('checkpoint_mtime', -1) - ckpt_mtime) < 1):
                all_log_probs = [data[f'lp_{i}'] for i in range(len(ids))]
                print(f'Loaded {len(all_log_probs)} cached log-prob arrays from {cache_path}')
            else:
                print(f'Cache at {cache_path} does not match checkpoint/rows; recomputing')
        except Exception as exc:
            print(f'Cache load failed ({exc}); recomputing forward pass')

    if all_log_probs is None:
        all_log_probs = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc='Forward'):
            image = cv2.imread(str(Path(args.image_dir) / f"{row['ID']}.jpg"), cv2.IMREAD_GRAYSCALE)
            image = preprocess(image, config)
            x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
            with torch.inference_mode():
                logits = model(x)  # [T, 1, C]
            all_log_probs.append(F.log_softmax(logits[:, 0, :], dim=-1).cpu().numpy().astype(np.float32))
        if cache_path is not None:
            payload = {f'lp_{i}': lp for i, lp in enumerate(all_log_probs)}
            payload['meta'] = np.array([json.dumps(cache_meta)])
            np.savez_compressed(cache_path, **payload)
            print(f'Cached log-probs to {cache_path}')
    return all_log_probs


def write_predictions_csv(path, ids, refs, hyps):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'Target', 'Prediction'])
        w.writerows(zip(ids, refs, hyps))


def run_val(args, model, config, out_dir):
    val_df = pd.read_csv(args.val_csv, encoding='utf-8-sig')
    if args.limit:
        val_df = val_df.head(args.limit)
    refs = val_df['Target'].fillna('').astype(str).tolist()
    ids = val_df['ID'].astype(str).tolist()
    all_log_probs = get_log_probs(val_df, model, config, args, out_dir)

    beam_widths = [int(b) for b in args.beams.split(',') if b.strip() != '' and int(b) > 0]
    results = {}
    results_path = out_dir / 'beam_results.json'
    results_payload = {'checkpoint': args.checkpoint, 'results': results}
    for bw in beam_widths:
        hyps = [decode_logits(lp, config['charset'], bw) for lp in tqdm(all_log_probs, desc=f'beam={bw}')]
        cer_s, wer_s, final_s = compute_corpus_metrics(refs, hyps)
        results[bw] = {'cer': cer_s, 'wer': wer_s, 'final': final_s}
        print(f'beam={bw}: CER={cer_s:.4f} WER={wer_s:.4f} Final={final_s:.4f}')
        # persist incrementally so partial sweeps survive interruption
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_payload, f, indent=2)
        write_predictions_csv(out_dir / f'val_predictions_beam{bw}.csv', ids, refs, hyps)

    lm_results = {}
    if args.lm_path:
        lm = FusionLM.load(args.lm_path)
        print(f'Loaded fusion LM from {args.lm_path}')
        sweep = [float(x) for x in args.lm_weights.split(',') if x != ''] if args.lm_weights else [0.2]
        for lam in sweep:
            hyps = [decode_logits(lp, config['charset'], args.lm_beam, lm=lm, lm_weight=lam)
                    for lp in tqdm(all_log_probs, desc=f'lm lambda={lam}')]
            cer_s, wer_s, final_s = compute_corpus_metrics(refs, hyps)
            lm_results[lam] = {'cer': cer_s, 'wer': wer_s, 'final': final_s, 'beam': args.lm_beam}
            print(f'lambda={lam}: CER={cer_s:.4f} WER={wer_s:.4f} Final={final_s:.4f}')
            write_predictions_csv(out_dir / f'val_predictions_lm_lam{lam}.csv', ids, refs, hyps)
        if lm_results:
            best_lam = min(lm_results, key=lambda k: lm_results[k]['final'])
            results_payload['lm_sweep'] = {
                'lm_path': args.lm_path, 'beam': args.lm_beam,
                'results': {str(k): v for k, v in lm_results.items()},
                'best_lambda': best_lam, 'best_final': lm_results[best_lam]['final'],
            }
            print(f"Best lambda: {best_lam} (Final={lm_results[best_lam]['final']:.4f})")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results_payload, f, indent=2)
    print(f'Wrote results to {out_dir}')


def run_test(args, model, config, out_dir):
    test_df = pd.read_csv(args.test_csv, encoding='utf-8-sig')
    if args.limit:
        test_df = test_df.head(args.limit)  # smoke tests only; partial submission
    ids = test_df['ID'].astype(str).tolist()
    all_log_probs = get_log_probs(test_df, model, config, args, out_dir)

    lm = FusionLM.load(args.lm_path) if (args.lm_path and args.lm_weight != 0.0) else None
    hyps = [decode_logits(lp, config['charset'], args.beam, lm=lm, lm_weight=args.lm_weight)
            for lp in tqdm(all_log_probs, desc=f"test beam={args.beam} lam={args.lm_weight}")]
    empty = sum(1 for h in hyps if not h.strip())
    out_path = Path(args.submission_out) if args.submission_out else out_dir / 'submission_beam.csv'
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'Prediction'])
        w.writerows(zip(ids, hyps))
    print(f'Wrote submission to {out_path}: {len(hyps)} rows, {empty} empty predictions')
    with open(out_dir / 'test_decode_config.json', 'w', encoding='utf-8') as f:
        json.dump({'checkpoint': args.checkpoint, 'beam': args.beam,
                   'lm_path': args.lm_path, 'lm_weight': args.lm_weight,
                   'rows': len(hyps), 'empty': empty}, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mode', choices=['val', 'test'], default='val')
    parser.add_argument('--val-csv', default=str(Path('data') / 'splits' / 'val.csv'))
    parser.add_argument('--test-csv', default='data/Test.csv')
    parser.add_argument('--image-dir', default='data/images')
    parser.add_argument('--beams', default='1,5,10,25,50', help='val mode: comma-separated beam widths to benchmark')
    parser.add_argument('--beam', type=int, default=10, help='test mode / LM sweep beam width')
    parser.add_argument('--lm-path', default='', help='fusion LM pickle (from lm_ngram.py); enables LM fusion')
    parser.add_argument('--lm-beam', type=int, default=10, help='beam width used during the lambda sweep')
    parser.add_argument('--lm-weights', default='',
                        help='val mode: comma-separated lambda sweep, e.g. 0,0.05,0.1,0.2,0.3,0.5')
    parser.add_argument('--lm-weight', type=float, default=0.0, help='test mode: lambda for LM fusion')
    parser.add_argument('--submission-out', default='', help='test mode: submission CSV path')
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=0, help='0 = all rows (for quick smoke tests)')
    parser.add_argument('--no-cache', action='store_true', help='disable the log-softmax cache (forces forward recompute)')
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, config = load_model(args.checkpoint)
    if args.mode == 'test':
        run_test(args, model, config, out_dir)
    else:
        run_val(args, model, config, out_dir)


if __name__ == '__main__':
    main()
