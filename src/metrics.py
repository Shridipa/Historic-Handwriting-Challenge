import math
from typing import Iterable, List, Sequence, Tuple


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insertions = curr[j - 1] + 1
            deletions = prev[j] + 1
            substitutions = prev[j - 1] + (ca != cb)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    if reference == hypothesis:
        return 0.0
    ref_len = max(len(reference), 1)
    return _levenshtein_distance(reference, hypothesis) / ref_len


def wer(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if reference == hypothesis:
        return 0.0
    if not ref_tokens and not hyp_tokens:
        return 0.0
    ref_len = max(len(ref_tokens), 1)
    return _levenshtein_distance(' '.join(ref_tokens), ' '.join(hyp_tokens)) / ref_len


def compute_corpus_metrics(references: Sequence[str], hypotheses: Sequence[str]) -> Tuple[float, float, float]:
    if len(references) != len(hypotheses):
        raise ValueError('Reference and hypothesis lengths differ')

    total_ref_chars = 0
    total_edit_chars = 0
    total_ref_words = 0
    total_edit_words = 0

    for ref, hyp in zip(references, hypotheses):
        ref = ref or ''
        hyp = hyp or ''
        total_ref_chars += len(ref)
        total_edit_chars += _levenshtein_distance(ref, hyp)

        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        total_ref_words += max(len(ref_tokens), 1)
        total_edit_words += _levenshtein_distance(' '.join(ref_tokens), ' '.join(hyp_tokens))

    cer_score = total_edit_chars / total_ref_chars if total_ref_chars else 0.0
    wer_score = total_edit_words / total_ref_words if total_ref_words else 0.0
    final = 0.7 * cer_score + 0.3 * wer_score
    return cer_score, wer_score, final


def final_score(cer_score: float, wer_score: float) -> float:
    return 0.7 * cer_score + 0.3 * wer_score


def normalize_text(text: str) -> str:
    return ' '.join((text or '').strip().split())
