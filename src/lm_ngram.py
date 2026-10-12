"""N-gram language models for CTC beam-search fusion (Phase 11).

DATA POLICY (challenge rule): these LMs are trained ONLY on transcripts from
data/splits/train.csv. Validation transcripts are used exclusively for tuning
the LM weight lambda -- never for training. Test transcripts are never touched.

Provides:
  CharNGramLM  -- order-n char model with stupid backoff (scores every emitted char)
  WordNGramLM  -- word uni/bigram model (scores completed space-delimited words)
  FusionLM     -- combination exposed through one incremental interface:
                  score_transition(prev_text, next_char) and final_score(text)
"""
import csv
import json
import math
import pickle
from pathlib import Path

BACKOFF = 0.4  # stupid-backoff factor (in linear domain)


def _load_texts(csv_path: str) -> list:
    texts = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            texts.append(row['Target'].strip())
    return texts


class CharNGramLM:
    """Order-n character model, stupid backoff, counts over the training texts."""

    def __init__(self, texts, order: int = 6, add_k: float = 0.05):
        self.order = order
        self.add_k = add_k
        # counts[k] = {context_tuple(len k-1): {char: count}} for k=1..order
        # plain dicts (not defaultdict) so the LM pickles cleanly
        self.counts = [None] + [{} for _ in range(order)]
        self.totals = [None] + [{} for _ in range(order)]
        for text in texts:
            seq = ['<s>'] * (order - 1) + list(text) + ['</s>']
            for k in range(1, order + 1):
                ctx_counts = self.counts[k]
                ctx_totals = self.totals[k]
                for i in range(len(seq) - k + 1):
                    ctx = tuple(seq[i:i + k - 1])
                    ch = seq[i + k - 1]
                    row = ctx_counts.get(ctx)
                    if row is None:
                        row = {}
                        ctx_counts[ctx] = row
                    row[ch] = row.get(ch, 0) + 1
                    ctx_totals[ctx] = ctx_totals.get(ctx, 0) + 1
        # unigram with add-k over observed vocab (floor for unseen chars)
        self.vocab = set(self.counts[1][()].keys())
        self.vocab_size = len(self.vocab)
        uni = self.counts[1][()]
        total_uni = sum(uni.values()) + add_k * self.vocab_size
        self._log_uni = {ch: math.log((c + add_k) / total_uni) for ch, c in uni.items()}
        self._log_uni_floor = math.log(add_k / total_uni)

    def logp(self, context: str, ch: str) -> float:
        """log P(ch | last (order-1) chars of context), stupid backoff."""
        seq = ['<s>'] * (self.order - 1) + list(context) + [ch]
        score = 0.0
        for k in range(self.order, 0, -1):
            ctx_k = tuple(seq[-k:-1])  # last k-1 symbols = context for order-k lookup
            ch_k = seq[-1]
            row = self.counts[k].get(ctx_k)
            total = self.totals[k].get(ctx_k, 0)
            if row is not None and total > 0 and ch_k in row:
                return score + math.log(row[ch_k] / total)
            score += math.log(BACKOFF)
        # fell through to unigram: add-k smoothed floor
        return score + self._log_uni.get(ch, self._log_uni_floor)


class WordNGramLM:
    """Word bigram LM with stupid backoff to add-k unigram; scores completed words."""

    def __init__(self, texts, add_k: float = 0.5):
        self.add_k = add_k
        self.bigrams = {}          # {prev_word: {word: count}} (plain dict: picklable)
        self.bigram_totals = {}
        self.unigrams = {}
        self.total_words = 0
        self._bos = '<s>'
        for text in texts:
            words = text.split()
            if not words:
                continue
            prev = self._bos
            for w in words:
                row = self.bigrams.setdefault(prev, {})
                row[w] = row.get(w, 0) + 1
                self.bigram_totals[prev] = self.bigram_totals.get(prev, 0) + 1
                self.unigrams[w] = self.unigrams.get(w, 0) + 1
                self.total_words += 1
                prev = w
            # sentence end marker
            row = self.bigrams.setdefault(prev, {})
            row['</s>'] = row.get('</s>', 0) + 1
            self.bigram_totals[prev] = self.bigram_totals.get(prev, 0) + 1
        self.vocab_size = len(self.unigrams) + 1  # +1 for </s>

    def logp_word(self, prev_word: str, word: str) -> float:
        total_bi = self.bigram_totals.get(prev_word, 0)
        if total_bi > 0 and word in self.bigrams.get(prev_word, {}):
            return math.log(self.bigrams[prev_word][word] / total_bi)
        # backoff to add-k unigram
        total_uni = self.total_words + self.add_k * self.vocab_size
        cnt = self.unigrams.get(word, 0)
        if word == '</s>':
            cnt = self.unigrams.get(word, 0)  # rare; treat as unseen unigram
            return math.log(BACKOFF) + math.log(self.add_k / total_uni)
        return math.log(BACKOFF) + math.log((cnt + self.add_k) / total_uni)

    def word_of(self, partial_text: str) -> str:
        """The word currently being built = text after the last space."""
        return partial_text.rsplit(' ', 1)[-1] if ' ' in partial_text else partial_text

    def prev_word_of(self, partial_text: str) -> str:
        """The last completed word, or <s> if none."""
        if ' ' not in partial_text:
            return self._bos
        completed = partial_text.rsplit(' ', 1)[0].rsplit(' ', 1)[-1]
        return completed if completed else self._bos


class FusionLM:
    """Uniform interface over char + word LMs for beam-search fusion.

    score_transition(prev_text, next_char):
      char LM  : log P(next_char | context)  (always, if char_lm set)
      word LM  : when next_char == ' ', log P(completed_word | prev_completed_word)
    final_score(text): word-LM score for the trailing (never space-terminated) word + EOS.
    All values are natural-log; the caller multiplies by lambda.
    """

    def __init__(self, char_lm=None, word_lm=None):
        self.char_lm = char_lm
        self.word_lm = word_lm

    def score_transition(self, prev_text: str, next_char: str) -> float:
        score = 0.0
        if self.char_lm is not None:
            score += self.char_lm.logp(prev_text, next_char)
        if self.word_lm is not None and next_char == ' ' and prev_text:
            cur_word = self.word_lm.word_of(prev_text)
            if cur_word:
                prev_w = self.word_lm.prev_word_of(prev_text)
                score += self.word_lm.logp_word(prev_w, cur_word)
        return score

    def final_score(self, text: str) -> float:
        if self.word_lm is None:
            return 0.0
        cur_word = self.word_lm.word_of(text) if text else ''
        prev_w = self.word_lm.prev_word_of(text) if text else '<s>'
        if not cur_word:
            # text ends in space (or empty): score EOS after last completed word
            prev_w = text.rsplit(' ', 2)[-2] if text.count(' ') >= 1 else '<s>'
            return self.word_lm.logp_word(prev_w, '</s>')
        return self.word_lm.logp_word(prev_w, cur_word) + self.word_lm.logp_word(cur_word, '</s>')

    # ---- persistence ----
    def save(self, out_dir: Path):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'fusion_lm.pkl', 'wb') as f:
            pickle.dump({'char_lm': self.char_lm, 'word_lm': self.word_lm}, f, protocol=4)
        meta = {
            'char_order': self.char_lm.order if self.char_lm else None,
            'char_vocab': self.char_lm.vocab_size if self.char_lm else None,
            'word_vocab': self.word_lm.vocab_size if self.word_lm else None,
            'word_tokens': self.word_lm.total_words if self.word_lm else None,
        }
        with open(out_dir / 'lm_meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load(path: str) -> 'FusionLM':
        class _RemapUnpickler(pickle.Unpickler):
            # LMs saved while lm_ngram.py ran as __main__ store classes under
            # __main__; remap them to this module so loads work everywhere.
            def find_class(self, module, name):
                if module == '__main__' and name in ('CharNGramLM', 'WordNGramLM', 'FusionLM'):
                    module = __name__
                return super().find_class(module, name)
        with open(path, 'rb') as f:
            d = _RemapUnpickler(f).load()
        return FusionLM(char_lm=d['char_lm'], word_lm=d['word_lm'])


def train_lm(train_csv: str, char_order: int = 6, use_char: bool = True, use_word: bool = True) -> FusionLM:
    """Train on data/splits/train.csv ONLY (see module docstring for data policy)."""
    texts = _load_texts(train_csv)
    char_lm = CharNGramLM(texts, order=char_order) if use_char else None
    word_lm = WordNGramLM(texts) if use_word else None
    return FusionLM(char_lm=char_lm, word_lm=word_lm)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', default=str(Path('data') / 'splits' / 'train.csv'))
    parser.add_argument('--out', default=str(Path('runs') / 'lm'))
    parser.add_argument('--char-order', type=int, default=6)
    args = parser.parse_args()

    lm = train_lm(args.train_csv, char_order=args.char_order)
    lm.save(Path(args.out))
    print(f'Trained LM on {args.train_csv} -> {args.out}')
    print(f"  char LM order={lm.char_lm.order} vocab={lm.char_lm.vocab_size}")
    print(f"  word LM vocab={lm.word_lm.vocab_size} tokens={lm.word_lm.total_words}")
    # sanity: high-prob continuations
    demo = lm.score_transition('In witness whereof the said', ' ')
    print(f'  score_transition("In witness whereof the said", " ") = {demo:.3f}')


if __name__ == '__main__':
    main()
