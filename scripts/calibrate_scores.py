"""Calibration script to determine optimal RAG and Chat grounding thresholds.

Evaluates cosine similarity distributions of on-topic versus off-topic query/chunk pairs
using the real local embedding model (fastembed ONNX, 100% offline, zero Gemini calls).
Prints score distributions and suggests CHAT_MIN_SCORE and RAG_MIN_SCORE thresholds.

Usage:
    python scripts/calibrate_scores.py
    python -m scripts.calibrate_scores --data tests/data/calibration_samples.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rag.embed import FastEmbedEmbedder, DEFAULT_CACHE_DIR, DEFAULT_EMBEDDING_MODEL

DEFAULT_SAMPLES_PATH = PROJECT_ROOT / "tests" / "data" / "calibration_samples.json"


def load_samples(path: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Load calibration sample pairs from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Calibration data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("on_topic", []), data.get("off_topic", [])


def compute_pair_scores(embedder: FastEmbedEmbedder, pairs: List[Dict[str, str]]) -> List[float]:
    """Embed query and chunk for each pair and return cosine similarity scores."""
    if not pairs:
        return []

    queries = [p["query"] for p in pairs]
    chunks = [p["chunk"] for p in pairs]

    q_vecs = np.asarray(embedder.embed_texts(queries), dtype=np.float32)
    c_vecs = np.asarray(embedder.embed_texts(chunks), dtype=np.float32)

    # Unit normalization check
    q_norms = np.linalg.norm(q_vecs, axis=1, keepdims=True)
    c_norms = np.linalg.norm(c_vecs, axis=1, keepdims=True)
    q_normed = q_vecs / np.maximum(q_norms, 1e-9)
    c_normed = c_vecs / np.maximum(c_norms, 1e-9)

    # Row-wise dot product of normalized vectors = cosine similarity
    scores = np.sum(q_normed * c_normed, axis=1)
    return [float(s) for s in scores]


def summarize_distribution(name: str, scores: List[float]) -> Dict[str, float]:
    """Calculate descriptive statistics and percentiles for a score array."""
    if not scores:
        return {}
    arr = np.array(scores)
    return {
        "count": len(arr),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def format_stats_table(stats_on: Dict[str, float], stats_off: Dict[str, float]) -> str:
    """Format comparison table of on-topic vs off-topic distributions."""
    header = f"{'Metric':<12} | {'On-Topic (Relevant)':<22} | {'Off-Topic (Noise)':<22}"
    divider = "-" * len(header)
    rows = [divider, header, divider]

    keys = [
        ("Count", "count", "{:d}"),
        ("Mean", "mean", "{:.4f}"),
        ("Std Dev", "std", "{:.4f}"),
        ("Min", "min", "{:.4f}"),
        ("Max", "max", "{:.4f}"),
        ("p05", "p05", "{:.4f}"),
        ("p10", "p10", "{:.4f}"),
        ("Median (p50)", "p50", "{:.4f}"),
        ("p90", "p90", "{:.4f}"),
        ("p95", "p95", "{:.4f}"),
    ]

    for label, k, fmt in keys:
        on_val = fmt.format(int(stats_on[k])) if k == "count" else fmt.format(stats_on[k])
        off_val = fmt.format(int(stats_off[k])) if k == "count" else fmt.format(stats_off[k])
        rows.append(f"{label:<12} | {on_val:<22} | {off_val:<22}")

    rows.append(divider)
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Calibrate RAG and Chat grounding score thresholds.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_SAMPLES_PATH), help="Path to JSON samples file.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Path to fastembed model cache directory.")
    parser.add_argument("--model", type=str, default=None, help="Embedding model name.")
    args = parser.parse_args()

    cache_dir = args.cache_dir or os.environ.get("EMBEDDING_CACHE_DIR", DEFAULT_CACHE_DIR)
    model_name = args.model or os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    print("==================================================================")
    print("RAG & Chat Grounding Score Calibration")
    print(f"Model:     {model_name}")
    print(f"Cache Dir: {cache_dir}")
    print(f"Data:      {args.data}")
    print("==================================================================")

    try:
        embedder = FastEmbedEmbedder(model_name=model_name, cache_dir=cache_dir)
    except Exception as exc:
        print(f"\n[ERROR] Failed to load local embedding model: {exc}")
        print("\nPlease run `python -m core.rag.download_model` first to download weights.")
        sys.exit(1)

    on_topic_pairs, off_topic_pairs = load_samples(Path(args.data))
    print(f"Loaded {len(on_topic_pairs)} on-topic and {len(off_topic_pairs)} off-topic pairs.\n")

    on_scores = compute_pair_scores(embedder, on_topic_pairs)
    off_scores = compute_pair_scores(embedder, off_topic_pairs)

    stats_on = summarize_distribution("On-Topic", on_scores)
    stats_off = summarize_distribution("Off-Topic", off_scores)

    print(format_stats_table(stats_on, stats_off))

    off_p95 = stats_off["p95"]
    on_p10 = stats_on["p10"]
    gap = on_p10 - off_p95

    # Suggested midpoint rounded to 2 decimal places
    suggested = round(off_p95 + gap / 2.0, 2) if gap > 0 else 0.35

    print("\nDistribution Analysis:")
    print(f"  - Off-Topic p95 (Noise upper ceiling):  {off_p95:.4f}")
    print(f"  - On-Topic  p10 (Grounding lower floor): {on_p10:.4f}")
    print(f"  - Clean Discrimination Margin (gap):    {gap:+.4f}")

    if gap > 0:
        print("\n[Recommendation]")
        print("  A clean separation exists between relevant evidence and off-topic noise.")
        print(f"  Suggested RAG_MIN_SCORE:  {suggested:.2f} (or conservative {round(off_p95, 2):.2f})")
        print(f"  Suggested CHAT_MIN_SCORE: {suggested:.2f} (default: 0.35)")
    else:
        print("\n[Warning] Overlap observed between off-topic noise and on-topic evidence.")
        print("  Suggested RAG_MIN_SCORE:  0.35")
        print("  Suggested CHAT_MIN_SCORE: 0.35")

    print("==================================================================\n")


if __name__ == "__main__":
    main()
