# Historic Handwriting Challenge

This project is a handwritten-text-recognition pipeline for the Barbados Historic Handwriting Challenge. It uses a CRNN-based OCR model trained with CTC loss, adaptive image preprocessing, and the challenge's metric definitions to evaluate recognition quality on historical manuscript pages.

The codebase is designed to run on CPU-only systems and is organized around reproducible training, validation, and test-prediction workflows.

## Project goal

The goal is to transcribe historical handwritten text from scanned documents into readable text while staying within the challenge evaluation rules. The repository includes:

- data preparation and splitting
- CRNN training and checkpointing
- validation metric reporting
- inference for the test set
- submission file generation
- optional beam-search + n-gram language model decoding for better transcription quality

## Repository structure

```text
.
├── data/
│   ├── Train.csv
│   ├── Test.csv
│   ├── SampleSubmission.csv
│   └── images/
├── runs/
│   ├── baseline/
│   ├── convergence_h80_cont/
│   ├── prep_sharp_h80_cont/
│   └── ...
├── src/
│   ├── train_crnn.py
│   ├── infer_crnn.py
│   ├── metrics.py
│   ├── validate_submission.py
│   ├── beam_search.py
│   └── ...
├── requirements.txt
├── submission.csv
├── submission_baseline_best.csv
├── README.md
└── requirements.txt
```

The challenge dataset, image archives, model checkpoints, and generated experiment files are kept locally and excluded from Git. Obtain the dataset from the challenge source and place its CSV files and images in `data/` before running the training or inference commands below. Detailed run logs and model outputs remain local under `runs/`.

## Setup

Create and activate a local Python environment, then install the dependencies.

```bash
cd "C:/Users/KIIT/Downloads/Historic Handwriting Challenge"
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data preparation

Prepare the dataset and cache any derived artifacts.

```bash
python src/train_crnn.py --prepare-only
```

## Training

A baseline CRNN run can be started with:

```bash
python src/train_crnn.py --experiment crnn_cpu_baseline --epochs 8 --batch-size 8 --lr 0.001 --val-split 0.1
```

More advanced runs can use the project's adaptive preprocessing and longer experiment configs under `runs/`.

## Validation and inference

Validate the generated submission file against the official test rows:

```bash
python src/validate_submission.py --submission submission.csv --test-csv data/Test.csv
```

Run test-time inference with a trained checkpoint:

```bash
python src/infer_crnn.py --checkpoint runs/<experiment>/checkpoints/best.pt --output submission.csv
```

Example for a checkpoint trained in the project:

```bash
python src/infer_crnn.py --checkpoint runs/convergence_h80_cont/checkpoints/best.pt --output submission.csv
```

## Challenge metric

The project uses the same weighted objective as the challenge:

```text
final = 0.7 * CER + 0.3 * WER
```

Where:

- CER = Character Error Rate
- WER = Word Error Rate
- final = weighted score used for comparison

This logic is implemented in `src/metrics.py`.

## Current validation results

The repository contains validated project outputs and the recorded best operating point from the challenge runs.

| Run / configuration | CER | WER | Final |
|---|---:|---:|---:|
| Baseline reproduction | 0.1142 | 0.6286 | 0.2685 |
| Converged CRNN checkpoint | 0.1062 | 0.5852 | 0.2458 |
| Best LM + beam-search operating point | 0.0733 | 0.4023 | 0.1720 |

The final tuned operating point is the best recorded result in the project report and is the result used for the challenge-level comparison.

## Submission status

The checked-in `submission.csv` is structurally valid for this project and passes the validator:

```bash
python src/validate_submission.py --submission submission.csv --test-csv data/Test.csv
```

Expected output:

```text
PASS
```

## Notes

- The project is intentionally built for CPU-safe execution.
- Training and validation outputs are stored under `runs/`.
- The best-performing configuration combines CRNN decoding with a language-model-aware beam-search pass.
- `report.md` contains the project's recorded benchmark summary and run history.

## License and usage

This repository is intended for research and challenge reproduction purposes. It is not a production OCR service and should be used as a reproducible baseline for historical handwriting recognition experiments.
