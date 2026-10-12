#!/usr/bin/env python3
"""Merge independently evaluated LoCoMo root shards with strict checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUM_FIELDS = (
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_prompt_tokens",
    "cache_write_prompt_tokens",
    "cached_prompt_detail_calls",
    "cache_write_prompt_detail_calls",
    "uncached_prompt_tokens",
    "uncached_plus_completion_tokens",
)


def add_bucket(dst: dict[str, int], src: dict[str, Any]) -> None:
    values = dict(src)
    if "uncached_prompt_tokens" not in values:
        prompt = int(values.get("prompt_tokens", 0) or 0)
        cached = int(values.get("cached_prompt_tokens", 0) or 0)
        values["uncached_prompt_tokens"] = max(0, prompt - cached)
    if "uncached_plus_completion_tokens" not in values:
        values["uncached_plus_completion_tokens"] = (
            int(values.get("uncached_prompt_tokens", 0) or 0)
            + int(values.get("completion_tokens", 0) or 0)
        )
    for field in SUM_FIELDS:
        dst[field] = int(dst.get(field, 0)) + int(values.get(field, 0) or 0)


def merge_token_summaries(summaries: list[dict[str, Any]], question_count: int) -> dict[str, Any]:
    stages: dict[str, dict[str, int]] = {}
    for summary in summaries:
        for name, bucket in (summary.get("by_stage") or {}).items():
            target = stages.setdefault(name, {field: 0 for field in SUM_FIELDS})
            add_bucket(target, bucket)

    def totals(prefixes: tuple[str, ...]) -> dict[str, int]:
        result = {field: 0 for field in SUM_FIELDS}
        for name, bucket in stages.items():
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                add_bucket(result, bucket)
        return result

    maintenance = totals(("maintenance",))
    retrieval = totals(("retrieval",))
    answer = totals(("answer",))
    judge = totals(("judge",))
    online = {
        field: retrieval[field] + answer[field]
        for field in SUM_FIELDS
    }
    total = {
        field: maintenance[field] + online[field]
        for field in SUM_FIELDS
    }
    return {
        "definition": {
            "candidate_online_token_cost": "retrieval + answer; maintenance/build excluded",
            "candidate_total_token_cost": "maintenance + retrieval + answer",
            "token_units": "prompt + completion tokens",
            "exactness": "preserved from shard API usage records",
            "uncached_plus_completion_token_view": (
                "prompt tokens minus reported cached prompt tokens, plus completion tokens; "
                "detail-call counts expose missing cache annotations. "
                "cached tokens are still billed under provider cache pricing, so this is not a monetary cost"
            ),
        },
        "shards_merged": len(summaries),
        "by_stage": stages,
        "maintenance": maintenance,
        "retrieval": retrieval,
        "answer": answer,
        "judge": judge,
        "candidate_online_token_cost": online,
        "candidate_total_token_cost": total,
        "candidate_online_uncached_plus_completion": {
            field: online[field]
            for field in ("calls", "cached_prompt_detail_calls", "cache_write_prompt_detail_calls", "uncached_prompt_tokens", "completion_tokens", "uncached_plus_completion_tokens")
        },
        "candidate_total_uncached_plus_completion": {
            field: total[field]
            for field in ("calls", "cached_prompt_detail_calls", "cache_write_prompt_detail_calls", "uncached_prompt_tokens", "completion_tokens", "uncached_plus_completion_tokens")
        },
        "question_count": question_count,
        "candidate_online_average_total_tokens_per_question": (
            online["total_tokens"] / question_count if question_count else None
        ),
        "candidate_total_average_tokens_per_question": (
            total["total_tokens"] / question_count if question_count else None
        ),
        "candidate_online_average_uncached_plus_completion_per_question": (
            online["uncached_plus_completion_tokens"] / question_count if question_count else None
        ),
        "candidate_total_average_uncached_plus_completion_per_question": (
            total["uncached_plus_completion_tokens"] / question_count if question_count else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")

    shard_payloads: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    seen_roots: set[int] = set()
    seen_questions: set[tuple[int, int]] = set()
    for shard_index in range(args.num_shards):
        shard_dir = args.run_root / f"locomo_{args.array_job_id}_{shard_index}"
        result_path = shard_dir / "results.json"
        token_path = shard_dir / "token_usage_summary.json"
        marker = shard_dir / "MEMWIRE_COMPLETED"
        missing = [str(path) for path in (result_path, token_path, marker) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Shard {shard_index} is incomplete; missing: {', '.join(missing)}"
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        summary = payload.get("summary") or {}
        if int(summary.get("shard_index", -1)) != shard_index:
            raise ValueError(f"Shard metadata mismatch in {result_path}")
        if int(summary.get("num_shards", -1)) != args.num_shards:
            raise ValueError(f"Shard count mismatch in {result_path}")
        roots = [int(index) for index in summary.get("root_indices", [])]
        if roots != [shard_index]:
            raise ValueError(f"Expected root [{shard_index}], got {roots} in {result_path}")
        rows = payload.get("detailed_results") or []
        if not rows:
            raise ValueError(f"Shard {shard_index} has no detailed results")
        for row in rows:
            key = (int(row["sample_idx"]), int(row["question_idx"]))
            if key in seen_questions:
                raise ValueError(f"Duplicate question key {key}")
            seen_questions.add(key)
            seen_roots.add(key[0])
        token_summary = json.loads(token_path.read_text(encoding="utf-8"))
        shard_payloads.append((shard_index, payload, token_summary))

    if seen_roots != set(range(args.num_shards)):
        raise ValueError(f"Root coverage mismatch: observed {sorted(seen_roots)}")

    rows = [row for _, payload, _ in shard_payloads for row in payload["detailed_results"]]
    rows.sort(key=lambda row: (int(row["sample_idx"]), int(row["question_idx"])))
    merged_tokens = merge_token_summaries(
        [token_summary for _, _, token_summary in shard_payloads], len(rows)
    )
    root_summaries = [payload.get("summary", {}) for _, payload, _ in shard_payloads]
    merged_summary = {
        "num_samples": args.num_shards,
        "shard_index": None,
        "num_shards": 1,
        "root_indices": sorted(seen_roots),
        "num_questions": len(rows),
        "source_array_job_id": str(args.array_job_id),
        "source_shards": [index for index, _, _ in shard_payloads],
        "token_usage": merged_tokens,
        "shard_summaries": root_summaries,
    }
    merged_payload = {
        "summary": merged_summary,
        "aggregated_metrics": {},
        "detailed_results": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(merged_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "token_usage_summary.json").write_text(
        json.dumps(merged_tokens, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "MERGE_COMPLETED").write_text(
        f"array_job_id={args.array_job_id}\nquestions={len(rows)}\n",
        encoding="utf-8",
    )
    print(f"Merged {args.num_shards} shards, {len(rows)} questions into {args.output_dir}")


if __name__ == "__main__":
    main()
