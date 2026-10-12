"""Pre-download TrOCR handwritten model + processor for Phase 9 (runs detached)."""
from huggingface_hub import snapshot_download

path = snapshot_download('microsoft/trocr-base-handwritten')
print(f'downloaded to {path}')
