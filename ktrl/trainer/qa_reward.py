"""Deterministic HotpotQA reward (normalized Token-F1 + format bonus, plus Exact Match)."""

from collections import Counter
import re
import string

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_EXCLUDE_PUNCT = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """Official HotpotQA normalization: lowercase, strip punctuation/articles, collapse spaces."""
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in _EXCLUDE_PUNCT)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def extract_answer(solution_str: str) -> tuple[str, bool]:
    """Extract final answer from <answer>...</answer>; fallback to \\boxed{} or last line."""
    text = (solution_str or "").strip()
    if not text:
        return "", False
    tag_matches = _ANSWER_TAG_RE.findall(text)
    if tag_matches:
        return tag_matches[-1].strip(), True
    boxed_matches = _BOXED_RE.findall(text)
    if boxed_matches:
        return boxed_matches[-1].strip(), False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (lines[-1] if lines else ""), False


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(ground_truth)
    if norm_pred in ("yes", "no", "noanswer") and norm_pred != norm_gold:
        return 0.0
    if norm_gold in ("yes", "no", "noanswer") and norm_pred != norm_gold:
        return 0.0
    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def compute_qa_score(solution_str: str, ground_truth: str) -> dict:
    """Compute deterministic HotpotQA score in [0.0, 1.0]: 0.9 * F1 + 0.1 * format_tag."""
    pred, has_format = extract_answer(solution_str)
    em = exact_match_score(pred, ground_truth)
    f1 = f1_score(pred, ground_truth)
    score = round(0.9 * f1 + (0.1 if has_format else 0.0), 6)
    return {
        "score": score,
        "acc": bool(em == 1.0),
        "em": em,
        "f1": round(f1, 6),
        "has_format": has_format,
        "pred": pred,
    }
