#!/usr/bin/env python3
"""Recompute transient-assembly metrics using the workbook contract."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


TABLE_CATEGORIES = (
    (1, "MultiHop"),
    (2, "Temporal"),
    (3, "OpenDomain"),
    (4, "SingleHop"),
)
ALL_CATEGORIES = (*TABLE_CATEGORIES, (5, "Adversarial"))


def normalize(text: Any) -> str:
    return str(text).lower()


def simple_tokens(text: Any) -> list[str]:
    return re.sub(r"[.,!?]", " ", normalize(text)).split()


def f1_score(prediction: Any, reference: Any) -> float:
    prediction_tokens = set(simple_tokens(prediction))
    reference_tokens = set(simple_tokens(reference))
    if not prediction_tokens or not reference_tokens:
        return 0.0
    common = len(prediction_tokens & reference_tokens)
    precision = common / len(prediction_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def bleu1_score(prediction: Any, reference: Any) -> float:
    prediction_tokens = word_tokenize(normalize(prediction))
    reference_tokens = [word_tokenize(normalize(reference))]
    return float(
        sentence_bleu(
            reference_tokens,
            prediction_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=SmoothingFunction().method1,
        )
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict[str, Any]], token_summary: dict[str, Any]) -> dict[str, Any]:
    buckets: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"f1": [], "bleu1": [], "stored_f1": [], "stored_bleu1": []}
    )
    for row in rows:
        category = int(row["category"])
        bucket = buckets[category]
        prediction = row.get("answer", "")
        reference = row.get("reference", "")
        bucket["f1"].append(f1_score(prediction, reference))
        bucket["bleu1"].append(bleu1_score(prediction, reference))
        stored = row.get("metrics", {})
        bucket["stored_f1"].append(float(stored.get("f1", 0.0)))
        bucket["stored_bleu1"].append(float(stored.get("bleu1", 0.0)))

    per_category: dict[str, dict[str, Any]] = {}
    category_names = dict((*TABLE_CATEGORIES, (5, "Adversarial")))
    for category, name in category_names.items():
        bucket = buckets[category]
        per_category[str(category)] = {
            "name": name,
            "count": len(bucket["f1"]),
            "f1": mean(bucket["f1"]),
            "bleu1": mean(bucket["bleu1"]),
            "max_abs_f1_recompute_diff": max(
                (abs(a - b) for a, b in zip(bucket["f1"], bucket["stored_f1"])),
                default=0.0,
            ),
            "max_abs_bleu1_recompute_diff": max(
                (abs(a - b) for a, b in zip(bucket["bleu1"], bucket["stored_bleu1"])),
                default=0.0,
            ),
        }

    four_category_f1 = mean([per_category[str(category)]["f1"] for category, _ in TABLE_CATEGORIES])
    four_category_bleu1 = mean([per_category[str(category)]["bleu1"] for category, _ in TABLE_CATEGORIES])
    online = token_summary.get("candidate_online_token_cost", {})
    total = token_summary.get("candidate_total_token_cost", {})
    uncached_online = token_summary.get("candidate_online_uncached_plus_completion", {})
    uncached_total = token_summary.get("candidate_total_uncached_plus_completion", {})
    # Older summaries predate provider cache accounting.  Keep their raw
    # numbers intact and derive a conservative uncached view when possible.
    if not uncached_online:
        cached = int(online.get("cached_prompt_tokens", 0) or 0)
        uncached = max(0, int(online.get("prompt_tokens", 0) or 0) - cached)
        uncached_online = {
            "calls": int(online.get("calls", 0) or 0),
            "cached_prompt_detail_calls": int(online.get("cached_prompt_detail_calls", 0) or 0),
            "cache_write_prompt_detail_calls": int(online.get("cache_write_prompt_detail_calls", 0) or 0),
            "uncached_prompt_tokens": uncached,
            "completion_tokens": int(online.get("completion_tokens", 0) or 0),
            "uncached_plus_completion_tokens": uncached + int(online.get("completion_tokens", 0) or 0),
        }
    if not uncached_total:
        cached = int(total.get("cached_prompt_tokens", 0) or 0)
        uncached = max(0, int(total.get("prompt_tokens", 0) or 0) - cached)
        uncached_total = {
            "calls": int(total.get("calls", 0) or 0),
            "cached_prompt_detail_calls": int(total.get("cached_prompt_detail_calls", 0) or 0),
            "cache_write_prompt_detail_calls": int(total.get("cache_write_prompt_detail_calls", 0) or 0),
            "uncached_prompt_tokens": uncached,
            "completion_tokens": int(total.get("completion_tokens", 0) or 0),
            "uncached_plus_completion_tokens": uncached + int(total.get("completion_tokens", 0) or 0),
        }
    question_count = len(rows)
    main_table = {
        name: {
            "f1": round(per_category[str(category)]["f1"] * 100, 2),
            "bleu": round(per_category[str(category)]["bleu1"] * 100, 2),
        }
        for category, name in TABLE_CATEGORIES
    }
    main_table["Average"] = {
        "f1": round(four_category_f1 * 100, 2),
        "bleu": round(four_category_bleu1 * 100, 2),
    }
    return {
        "questions": question_count,
        "category_mapping": {str(category): name for category, name in (*TABLE_CATEGORIES, (5, "Adversarial"))},
        "per_category": per_category,
        "main_table_percent": main_table,
        "macro_average_f1": four_category_f1,
        "macro_average_bleu1": four_category_bleu1,
        "adversarial_excluded_from_four_category_average": True,
        "token_cost": {
            "definition": "candidate_online_token_cost (retrieval + answer; maintenance excluded)",
            "calls": int(online.get("calls", 0) or 0),
            "prompt_tokens": int(online.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(online.get("completion_tokens", 0) or 0),
            "total_tokens": int(online.get("total_tokens", 0) or 0),
            "tokens_per_question": (int(online.get("total_tokens", 0) or 0) / question_count if question_count else 0.0),
            "cached_prompt_tokens": int(online.get("cached_prompt_tokens", 0) or 0),
            "cache_write_prompt_tokens": int(online.get("cache_write_prompt_tokens", 0) or 0),
            "cached_prompt_detail_calls": int(uncached_online.get("cached_prompt_detail_calls", 0) or 0),
            "cache_write_prompt_detail_calls": int(uncached_online.get("cache_write_prompt_detail_calls", 0) or 0),
            "uncached_prompt_tokens": int(uncached_online.get("uncached_prompt_tokens", 0) or 0),
            "uncached_plus_completion_tokens": int(uncached_online.get("uncached_plus_completion_tokens", 0) or 0),
            "uncached_plus_completion_tokens_per_question": (
                int(uncached_online.get("uncached_plus_completion_tokens", 0) or 0) / question_count
                if question_count else 0.0
            ),
            "maintenance_plus_online_total_tokens": int(total.get("total_tokens", 0) or 0),
            "maintenance_plus_online_uncached_plus_completion_tokens": int(
                uncached_total.get("uncached_plus_completion_tokens", 0) or 0
            ),
        },
    }


def render_markdown(report: dict[str, Any], model: str, method: str, elapsed: str) -> str:
    table = report["main_table_percent"]
    headers = ["Model", "Method", "Jop Num"]
    values = [model, method, str(report["questions"])]
    for _, name in TABLE_CATEGORIES:
        headers.extend((f"{name} F1", f"{name} BLEU"))
        values.extend((f"{table[name]['f1']:.2f}", f"{table[name]['bleu']:.2f}"))
    headers.extend(("Average F1", "Average BLEU", "Token Cost", "Local Time"))
    values.extend(
        (
            f"{table['Average']['f1']:.2f}",
            f"{table['Average']['bleu']:.2f}",
            f"{report['token_cost']['tokens_per_question']:.2f} tok/q",
            elapsed or "N/A",
        )
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        "| " + " | ".join(values) + " |",
        "",
        "Metric contract: F1 and smoothed BLEU-1; Average is the macro mean over categories 1-4.",
        "Category 5 (Adversarial) is reported separately and is excluded from that four-category Average.",
        "Raw Token Cost is online retrieval + answer prompt+completion tokens; maintenance/build tokens are excluded.",
        "Uncached-plus-completion subtracts reported cached prompt tokens from the raw total; "
        "cached tokens may still incur provider cache charges, so this is not a monetary cost.",
        (
            "Uncached-plus-completion: "
            f"{report['token_cost']['uncached_plus_completion_tokens_per_question']:.2f} tok/q."
        ),
    ]
    adversarial = report["per_category"]["5"]
    lines.append(
        f"Adversarial (n={adversarial['count']}): F1={adversarial['f1'] * 100:.2f}, "
        f"BLEU-1={adversarial['bleu1'] * 100:.2f}."
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--token-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--method", default="MemRewire")
    parser.add_argument("--elapsed", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_payload = json.loads(args.results.read_text(encoding="utf-8"))
    rows = result_payload.get("detailed_results", [])
    if not rows:
        raise ValueError(f"No detailed_results found in {args.results}")
    token_summary = json.loads(args.token_summary.read_text(encoding="utf-8"))
    report = summarize(rows, token_summary)
    report["model"] = args.model
    report["method"] = args.method
    report["local_time"] = args.elapsed
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "excel_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "excel_metrics.md").write_text(render_markdown(report, args.model, args.method, args.elapsed), encoding="utf-8")
    print(render_markdown(report, args.model, args.method, args.elapsed), end="")


if __name__ == "__main__":
    main()
