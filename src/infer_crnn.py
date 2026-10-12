import argparse
import csv
import json
from pathlib import Path

import cv2
import pandas as pd
import torch
from tqdm import tqdm

from data_utils import DATA_ROOT, preprocess_image, preprocess_image_aspect_preserving, preprocess_image_aspect_preserving_bgcontrast, preprocess_image_aspect_preserving_bgcontrast_sharp
from train_crnn import CRNN, decode_ctc
from metrics import compute_corpus_metrics, cer


def load_model(checkpoint_path):
    payload = torch.load(checkpoint_path, map_location='cpu')
    charset = payload['charset']
    model = CRNN(charset)
    model.load_state_dict(payload['model_state_dict'])
    model.eval()
    # Auto-detect the preprocessing configuration the checkpoint was trained with.
    # CLI flags can override, but defaults follow the checkpoint so that training
    # and inference preprocessing can never silently diverge.
    config = {
        'height': int(payload.get('height', 64)),
        'width': 768,
        'max_width': int(payload.get('max_width', 1536)),
        'adaptive_width': bool(payload.get('adaptive_width', False)),
        'preprocess_variant': str(payload.get('preprocess_variant', 'baseline')),
        'charset': charset,
        'epoch': payload.get('epoch'),
        'experiment': payload.get('experiment'),
    }
    return model, config


def preprocess(image, config):
    if config['adaptive_width']:
        if config['preprocess_variant'] == 'bgcontrast':
            return preprocess_image_aspect_preserving_bgcontrast(
                image, target_height=config['height'], max_width=config['max_width'])
        if config['preprocess_variant'] == 'bgcontrast_sharp':
            return preprocess_image_aspect_preserving_bgcontrast_sharp(
                image, target_height=config['height'], max_width=config['max_width'])
        return preprocess_image_aspect_preserving(
            image, target_height=config['height'], max_width=config['max_width'])
    return preprocess_image(image, target_height=config['height'], target_width=config['width'])


@torch.inference_mode()
def predict_df(df, model, config, image_dir, desc='Inference'):
    idx_to_char = {i + 1: ch for i, ch in enumerate(config['charset'])}
    preds = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        image_path = Path(image_dir) / f"{row['ID']}.jpg"
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f'Missing image {image_path}')
        image = preprocess(image, config)
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        logits = model(image)
        pred = decode_ctc(logits[:, 0, :], None, idx_to_char, max_len=256)
        preds.append(pred if pred else '')
    return preds


def run_validation(args, model, config):
    val_df = pd.read_csv(DATA_ROOT / 'data' / 'splits' / 'val.csv', encoding='utf-8-sig')
    hyps = predict_df(val_df, model, config, args.image_dir, desc='Validation')
    refs = val_df['Target'].fillna('').astype(str).tolist()
    cer_s, wer_s, final_s = compute_corpus_metrics(refs, hyps)
    exact = sum(r == h for r, h in zip(refs, hyps)) / max(len(refs), 1)
    char_acc = 1.0 - cer_s
    result = {
        'checkpoint': args.checkpoint,
        'experiment': config['experiment'],
        'epoch': config['epoch'],
        'preprocess_config': {k: v for k, v in config.items() if k != 'charset'},
        'val_cer': cer_s,
        'val_wer': wer_s,
        'val_final': final_s,
        'exact_match_accuracy': exact,
        'character_accuracy': char_acc,
        'n_val': len(refs),
        'empty_predictions': sum(1 for h in hyps if not h.strip()),
    }
    print(json.dumps(result, indent=2, default=str))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Target', 'Prediction', 'CER'])
        for row_id, ref, hyp in zip(val_df['ID'], refs, hyps):
            writer.writerow([row_id, ref, hyp, cer(ref, hyp)])
    print(f'Wrote validation predictions to {out_path}')
    result_path = out_path.parent / 'baseline_reproduction.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str)
    print(f'Wrote metrics to {result_path}')
    return result


def run_test(args, model, config):
    test_df = pd.read_csv(args.test_csv, encoding='utf-8-sig')
    preds = predict_df(test_df, model, config, args.image_dir, desc='Test')
    df = pd.DataFrame({'ID': test_df['ID'], 'Target': preds})
    df = df[['ID', 'Target']]
    df.to_csv(args.output, index=False, encoding='utf-8')
    print(f'Wrote {len(df)} predictions to {args.output}')
    print(f'Preprocessing config used: height={config["height"]} adaptive_width={config["adaptive_width"]} '
          f'max_width={config["max_width"]} variant={config["preprocess_variant"]}')


def main():
    parser = argparse.ArgumentParser(description='CRNN inference with checkpoint-matched preprocessing')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test-csv', default='data/Test.csv')
    parser.add_argument('--image-dir', default='data/images')
    parser.add_argument('--output', default='submission.csv')
    parser.add_argument('--mode', choices=['test', 'val'], default='test',
                        help='test: generate submission; val: score checkpoint on the validation split')
    # Optional overrides. By default the preprocessing configuration stored in the
    # checkpoint is used so inference always matches training.
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--adaptive-width', dest='adaptive_width', action='store_true', default=None)
    parser.add_argument('--no-adaptive-width', dest='adaptive_width', action='store_false')
    parser.add_argument('--max-width', type=int, default=None)
    parser.add_argument('--preprocess-variant', choices=['baseline', 'bgcontrast'], default=None)
    args = parser.parse_args()

    model, config = load_model(args.checkpoint)
    if args.height is not None:
        config['height'] = args.height
    if args.adaptive_width is not None:
        config['adaptive_width'] = args.adaptive_width
    if args.max_width is not None:
        config['max_width'] = args.max_width
    if args.preprocess_variant is not None:
        config['preprocess_variant'] = args.preprocess_variant

    if args.mode == 'val':
        run_validation(args, model, config)
    else:
        run_test(args, model, config)


if __name__ == '__main__':
    main()
