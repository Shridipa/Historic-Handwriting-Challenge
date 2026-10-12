import argparse
import csv
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_utils import (
    DATA_ROOT,
    IMG_ROOT,
    TRAIN_CSV,
    ensure_expected_layout,
    build_charset,
    preprocess_image,
    preprocess_image_aspect_preserving,
    preprocess_image_aspect_preserving_bgcontrast,
    preprocess_image_aspect_preserving_bgcontrast_sharp,
)
from metrics import cer, compute_corpus_metrics
from augment import augment_image, OPS as AUGMENT_OPS


def custom_collate(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    ids = [item[2] for item in batch]
    max_width = max(image.shape[-1] for image in images)
    images = [F.pad(image, (0, max_width - image.shape[-1])) for image in images]
    images = torch.stack(images, dim=0)
    padded_targets = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return images, padded_targets, target_lengths, ids


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class OCRDataset(Dataset):
    def __init__(self, df, charset, image_root, height=64, width=768, target_max_len=160, adaptive_width=False, max_width=1536, preprocess_variant='baseline', augment=()):
        self.df = df.reset_index(drop=True)
        self.charset = charset
        self.image_root = image_root
        self.height = height
        self.width = width
        self.target_max_len = target_max_len
        self.adaptive_width = adaptive_width
        self.max_width = max_width
        self.preprocess_variant = preprocess_variant
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(charset)}
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.augment = tuple(a for a in augment if a)
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.df)

    def _encode_text(self, text):
        chars = list(str(text))
        out = [self.char_to_idx.get(ch, 0) for ch in chars]
        if len(out) > self.target_max_len:
            out = out[: self.target_max_len]
        return torch.tensor(out, dtype=torch.long)

    def _preprocess(self, image):
        if self.adaptive_width:
            if self.preprocess_variant == 'bgcontrast':
                return preprocess_image_aspect_preserving_bgcontrast(image, target_height=self.height, max_width=self.max_width)
            if self.preprocess_variant == 'bgcontrast_sharp':
                return preprocess_image_aspect_preserving_bgcontrast_sharp(image, target_height=self.height, max_width=self.max_width)
            return preprocess_image_aspect_preserving(image, target_height=self.height, max_width=self.max_width)
        return preprocess_image(image, target_height=self.height, target_width=self.width)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = f"{row['ID']}.jpg"
        image = cv2.imread(str(self.image_root / img_name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f'Missing image: {self.image_root / img_name}')
        image = self._preprocess(image)
        if self.augment:
            image = augment_image(image, self.augment, self.rng)
        image = torch.from_numpy(image).unsqueeze(0).float()
        target = self._encode_text(row['Target'])
        return image, target, row['ID']


class CRNN(nn.Module):
    def __init__(self, charset, hidden_size=256, num_layers=2):
        super().__init__()
        self.charset = charset
        self.num_classes = len(charset) + 1
        self.blank_id = 0
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, self.num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, (1, x.size(-1)))
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.rnn(x)
        logits = self.classifier(x)
        return logits.permute(1, 0, 2)


def ctcloss(logits, targets, input_lengths, target_lengths):
    return nn.functional.ctc_loss(
        logits,
        targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        blank=self.blank_id,
        zero_infinity=True,
        reduction='mean',
    )


def decode_ctc(logits, char_to_idx, idx_to_char, max_len=160):
    probs = torch.softmax(logits, dim=-1)
    seq = torch.argmax(probs, dim=-1)
    pred = []
    prev = None
    for token in seq:
        if token.item() == 0:
            prev = None
            continue
        if token.item() == prev:
            continue
        pred.append(idx_to_char.get(token.item(), ''))
        prev = token.item()
    return ''.join(pred)


def calculate_metrics_for_split(model, loader, charset, device):
    model.eval()
    refs = []
    hyps = []
    idx_to_char = {i + 1: ch for i, ch in enumerate(charset)}
    with torch.no_grad():
        for images, targets, target_lengths, ids in loader:
            images = images.to(device)
            logits = model(images)
            for i in range(images.size(0)):
                seq = logits[:, i, :]
                pred = decode_ctc(seq, None, idx_to_char, max_len=256)
                ref = ''.join([charset[t - 1] for t in targets[i].tolist()[:target_lengths[i].item()] if t > 0])
                refs.append(ref)
                hyps.append(pred)
    cer_score, wer_score, final_score_value = compute_corpus_metrics(refs, hyps)
    exact_matches = sum(ref == hyp for ref, hyp in zip(refs, hyps))
    exact_match_accuracy = exact_matches / max(len(refs), 1)
    character_accuracy = 1.0 - cer_score
    summary = {
        'cer': cer_score,
        'wer': wer_score,
        'final': final_score_value,
        'exact_match_accuracy': exact_match_accuracy,
        'exact_match_accuracy_percent': exact_match_accuracy * 100.0,
        'character_accuracy': character_accuracy,
        'character_accuracy_percent': character_accuracy * 100.0,
    }
    return summary, refs, hyps


def train_model(args):
    ensure_expected_layout()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(DATA_ROOT / 'data' / 'Train.csv', encoding='utf-8-sig')
    charset = build_charset(train_df)
    if not args.charset:
        args.charset = charset
    char_to_idx = {ch: i + 1 for i, ch in enumerate(args.charset)}
    idx_to_char = {v: k for k, v in char_to_idx.items()}

    split_train = DATA_ROOT / 'data' / 'splits' / 'train.csv'
    split_val = DATA_ROOT / 'data' / 'splits' / 'val.csv'
    if split_train.exists() and split_val.exists():
        train_df = pd.read_csv(split_train, encoding='utf-8-sig')
        val_df = pd.read_csv(split_val, encoding='utf-8-sig')
    else:
        train_df, val_df = train_test_split(train_df, test_size=args.val_split, random_state=42, shuffle=True)
    train_df.to_csv(output_dir / 'train_split.csv', index=False, encoding='utf-8')
    val_df.to_csv(output_dir / 'val_split.csv', index=False, encoding='utf-8')

    # Explicit flag wins; fall back to legacy name-based detection for old commands.
    if getattr(args, 'preprocess_variant', 'auto') in ('auto', '', None):
        args.preprocess_variant = 'bgcontrast' if 'bgcontrast' in str(args.experiment).lower() else 'baseline'

    augment_ops = tuple(a.strip() for a in getattr(args, 'augment', '').split(',') if a.strip())
    for a in augment_ops:
        if a not in AUGMENT_OPS:
            raise ValueError(f'unknown augmentation op: {a!r}; choose from {AUGMENT_OPS}')
    train_ds = OCRDataset(train_df, args.charset, DATA_ROOT / 'data' / 'images', height=args.height, width=args.width, adaptive_width=args.adaptive_width, max_width=args.max_width, preprocess_variant=args.preprocess_variant, augment=augment_ops)
    val_ds = OCRDataset(val_df, args.charset, DATA_ROOT / 'data' / 'images', height=args.height, width=args.width, adaptive_width=args.adaptive_width, max_width=args.max_width, preprocess_variant=args.preprocess_variant)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate)

    config_lines = [
        f'experiment={args.experiment}',
        f'output_dir={args.output_dir}',
        f'epochs={args.epochs}',
        f'batch_size={args.batch_size}',
        f'height={args.height}',
        f'width={args.width}',
        f'val_split={args.val_split}',
        f'lr={args.lr}',
        f'adaptive_width={args.adaptive_width}',
        f'max_width={args.max_width}',
        f'preprocess_variant={args.preprocess_variant}',
        f'augment={getattr(args, "augment", "")}',
        f'init_checkpoint={args.init_checkpoint}',
        f'patience={args.patience}',
        f'charset_length={len(args.charset)}',
        f'cuda_available={torch.cuda.is_available()}',
    ]
    (output_dir / 'config.txt').write_text('\n'.join(config_lines) + '\n', encoding='utf-8')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(args.charset).to(device)
    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(payload['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = None
    best_epoch_seen = 0
    history = []
    best_path = Path(args.output_dir) / 'checkpoints' / 'best.pt'
    best_path.parent.mkdir(parents=True, exist_ok=True)
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, targets, target_lengths, ids in tqdm(train_loader, desc=f'Epoch {epoch}'):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=2)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long, device=device)
            loss = nn.functional.ctc_loss(
                log_probs,
                targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                blank=0,
                zero_infinity=True,
                reduction='mean',
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item() * images.size(0)

        metrics_summary, refs, hyps = calculate_metrics_for_split(model, val_loader, args.charset, device)
        history.append({
            'epoch': epoch,
            'train_loss': running / len(train_ds),
            'val_cer': float(metrics_summary['cer']),
            'val_wer': float(metrics_summary['wer']),
            'val_final': float(metrics_summary['final']),
            'exact_match_accuracy': float(metrics_summary['exact_match_accuracy']),
            'exact_match_accuracy_percent': float(metrics_summary['exact_match_accuracy_percent']),
            'character_accuracy': float(metrics_summary['character_accuracy']),
            'character_accuracy_percent': float(metrics_summary['character_accuracy_percent']),
        })
        epoch_path = output_dir / 'checkpoints' / f'epoch_{epoch:03d}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'charset': args.charset,
            'epoch': epoch,
            'metrics': {
                'cer': float(metrics_summary['cer']),
                'wer': float(metrics_summary['wer']),
                'final': float(metrics_summary['final']),
                'exact_match_accuracy': float(metrics_summary['exact_match_accuracy']),
                'character_accuracy': float(metrics_summary['character_accuracy']),
            },
            'experiment': args.experiment,
            'preprocess_variant': args.preprocess_variant,
            'height': args.height,
            'max_width': args.max_width,
            'adaptive_width': args.adaptive_width,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'val_split': args.val_split,
            'init_checkpoint': args.init_checkpoint,
        }, epoch_path)
        with open(output_dir / 'val_predictions.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Target', 'Prediction', 'CER'])
            for row_id, ref, hyp in zip([row_id for batch in val_loader for row_id in batch[3]], refs, hyps):
                writer.writerow([row_id, ref, hyp, cer(ref, hyp)])
        print(f"Epoch {epoch}: val CER={metrics_summary['cer']:.4f} WER={metrics_summary['wer']:.4f} FINAL={metrics_summary['final']:.4f} exact_match={metrics_summary['exact_match_accuracy']:.4f} char_acc={metrics_summary['character_accuracy']:.4f}")

        if best is None or metrics_summary['final'] < best:
            best = metrics_summary['final']
            best_epoch_seen = epoch
            patience_left = args.patience
            torch.save({
                'model_state_dict': model.state_dict(),
                'charset': args.charset,
                'epoch': epoch,
                'metrics': {
                    'cer': float(metrics_summary['cer']),
                    'wer': float(metrics_summary['wer']),
                    'final': float(metrics_summary['final']),
                    'exact_match_accuracy': float(metrics_summary['exact_match_accuracy']),
                    'character_accuracy': float(metrics_summary['character_accuracy']),
                },
                'experiment': args.experiment,
                'preprocess_variant': args.preprocess_variant,
                'height': args.height,
                'max_width': args.max_width,
                'adaptive_width': args.adaptive_width,
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'val_split': args.val_split,
                'init_checkpoint': args.init_checkpoint,
            }, best_path)
        else:
            patience_left -= 1

        metrics_payload = {
            'val_cer': float(metrics_summary['cer']),
            'val_wer': float(metrics_summary['wer']),
            'val_final': float(metrics_summary['final']),
            'exact_match_accuracy': float(metrics_summary['exact_match_accuracy']),
            'exact_match_accuracy_percent': float(metrics_summary['exact_match_accuracy_percent']),
            'character_accuracy': float(metrics_summary['character_accuracy']),
            'character_accuracy_percent': float(metrics_summary['character_accuracy_percent']),
            'best_epoch': best_epoch_seen,
            'history': history,
        }
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_payload, f, indent=2)
        if patience_left <= 0:
            print(f'Early stopping after epoch {epoch}')
            break

    final_metrics = {
        'val_cer': float(metrics_summary['cer']),
        'val_wer': float(metrics_summary['wer']),
        'val_final': float(metrics_summary['final']),
        'exact_match_accuracy': float(metrics_summary['exact_match_accuracy']),
        'exact_match_accuracy_percent': float(metrics_summary['exact_match_accuracy_percent']),
        'character_accuracy': float(metrics_summary['character_accuracy']),
        'character_accuracy_percent': float(metrics_summary['character_accuracy_percent']),
        'best_epoch': best_epoch_seen,
        'history': history,
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(json.dumps(final_metrics, indent=2))

    benchmark = {
        'cer': 0.12659961648338747,
        'wer': 0.6972576117469229,
        'final': 0.29779701506244804,
    }
    comparison_lines = [
        'Adaptive preprocessing robustness comparison',
        f'Experiment: {args.experiment}',
        f'Checkpoint: {best_path}',
        f'Current benchmark CER={benchmark["cer"]} WER={benchmark["wer"]} FINAL={benchmark["final"]}',
        f'New run CER={final_metrics["val_cer"]} WER={final_metrics["val_wer"]} FINAL={final_metrics["val_final"]}',
        '',
        'Relative change (new vs old):',
        f'CER delta={(final_metrics["val_cer"] - benchmark["cer"]):.6f}',
        f'WER delta={(final_metrics["val_wer"] - benchmark["wer"]):.6f}',
        f'FINAL delta={(final_metrics["val_final"] - benchmark["final"]):.6f}',
        f'CER relative improvement={((benchmark["cer"] - final_metrics["val_cer"]) / benchmark["cer"] * 100.0):.2f}%',
        f'WER relative improvement={((benchmark["wer"] - final_metrics["val_wer"]) / benchmark["wer"] * 100.0):.2f}%',
        f'FINAL relative improvement={((benchmark["final"] - final_metrics["val_final"]) / benchmark["final"] * 100.0):.2f}%',
    ]
    (output_dir / 'comparison_to_benchmark.txt').write_text('\n'.join(comparison_lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--experiment', default='crnn_cpu_baseline')
    parser.add_argument('--output-dir', default='runs/crnn_cpu_baseline')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--charset', default='')
    parser.add_argument('--init-checkpoint', default='')
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--adaptive-width', action='store_true')
    parser.add_argument('--max-width', type=int, default=1536)
    parser.add_argument('--preprocess-variant', choices=['auto', 'baseline', 'bgcontrast', 'bgcontrast_sharp'], default='auto')
    parser.add_argument('--augment', default='', help=f'comma-separated training augmentations from {AUGMENT_OPS}')
    args = parser.parse_args()
    if args.prepare_only:
        ensure_expected_layout()
        print('Prepared dataset layout.')
    else:
        train_model(args)
