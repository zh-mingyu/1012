"""Collect matched per-query evidence-access traces for graph variants."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any, Dict, Iterable, List

from memrewire.core.models.memory_entry import Dialogue
from memrewire.core.settings import settings as config
from memrewire_app import MemRewireSystem
from locomo_evidence import normalize_evidence_ids


CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}
GRAPH_MODES = {
    "initial": "initial",
    "memrewire": "operator-aware",
    "random": "random-rewiring",
}


def configure_logging(output_dir: Path) -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        output_dir / "debug.log", mode="a", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    logger = logging.getLogger("paired_graph")
    logger.setLevel(logging.INFO)
    return logger


def load_samples(dataset_path: Path, limit: int | None) -> List[Dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    return samples[:limit] if limit is not None else samples


def convert_to_dialogues(sample: Dict[str, Any]) -> List[Dialogue]:
    conversation = sample["conversation"]
    sessions = []
    for key, turns in conversation.items():
        if key.startswith("session_") and isinstance(turns, list):
            try:
                session_id = int(key.split("_")[1])
            except (IndexError, ValueError):
                continue
            sessions.append((session_id, turns, conversation.get(f"{key}_date_time")))

    dialogues: List[Dialogue] = []
    dialogue_id = 1
    for session_id, turns, timestamp in sorted(sessions):
        for turn_index, turn in enumerate(turns, 1):
            text = turn.get("text", "")
            if "img_url" in turn and "blip_caption" in turn:
                caption = f"[Image: {turn['blip_caption']}]"
                text = f"{caption} {text}" if text else caption
            dialogues.append(
                Dialogue(
                    dialogue_id=dialogue_id,
                    speaker=turn["speaker"],
                    content=text,
                    timestamp=timestamp,
                    source_id=turn["dia_id"],
                    session_id=f"D{session_id}",
                    turn_index=turn_index,
                )
            )
            dialogue_id += 1
    return dialogues


def category_name(value: Any) -> str:
    try:
        category = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return CATEGORY_NAMES.get(category, f"unknown-{category}")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2)
    temp_path.replace(path)


def append_jsonl(handle, data: Dict[str, Any]) -> None:
    handle.write(json.dumps(data, ensure_ascii=True) + "\n")
    handle.flush()


def graph_snapshot(
    threads: Iterable[Any],
    graph_name: str,
    sample_idx: int,
    run_id: str,
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    vertex_rows = []
    edge_rows = []
    for thread in sorted(threads, key=lambda item: item.thread_id):
        vertex_rows.append(
            {
                "cell_id": thread.thread_id,
                "source_turn_ids": sorted(
                    {
                        str(source_id)
                        for node in thread.nodes
                        for source_id in node.source_turn_ids
                    }
                ),
            }
        )
        for link in thread.links:
            if thread.thread_id < link.target_thread_id:
                edge_rows.append(
                    {
                        "source_cell_id": thread.thread_id,
                        "target_cell_id": link.target_thread_id,
                        "relation": link.relation,
                        "weight": float(link.weight),
                    }
                )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "sample_idx": sample_idx,
        "graph_name": graph_name,
        "vertices": vertex_rows,
        "edges": edge_rows,
        "graph_diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect retrieval-only paired graph evidence traces."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--graph-name", choices=sorted(GRAPH_MODES), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--readout-k", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=2026)
    args = parser.parse_args()

    if args.max_hops < 0:
        parser.error("--max-hops must be nonnegative")
    if args.readout_k < 1:
        parser.error("--readout-k must be positive")
    if args.num_samples is not None and args.num_samples < 1:
        parser.error("--num-samples must be positive")
    graph_mode = GRAPH_MODES[args.graph_name]
    config.THREAD_GRAPH_MODE = graph_mode
    config.THREAD_GRAPH_RANDOM_SEED = args.random_seed
    config.THREAD_GRAPH_RANDOM_SWAPS_PER_EDGE = 10
    config.ENABLE_THREAD_SEGMENTATION = False
    config.ENABLE_PARALLEL_PROCESSING = False
    config.RECEPTIVE_FIELD_HOPS = args.max_hops
    config.WORKSPACE_MAX_READOUT_CELLS = args.readout_k
    config.SOURCE_READOUT_K = args.readout_k
    config.WORKSPACE_READOUT_MODE = "query_similarity"
    config.THREAD_LANCEDB_PATH = str(args.output_dir / "thread_lancedb_data")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "retrieval_trace.jsonl"
    graph_path = args.output_dir / "graphs.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    completion_path = args.output_dir / "PAIRED_GRAPH_COMPLETED"
    if any(
        path.exists()
        for path in (
            trace_path,
            graph_path,
            manifest_path,
            completion_path,
            args.output_dir / "thread_lancedb_data",
        )
    ):
        parser.error("output directory already contains data; choose a new run ID")
    logger = configure_logging(args.output_dir)
    current_stage = {"name": "startup"}

    def mark_stage(name: str, **details: Any) -> None:
        current_stage["name"] = name
        detail_text = " ".join(f"{key}={value}" for key, value in details.items())
        logger.info("STAGE name=%s %s", name, detail_text)

    logger.info(
        "JOB_START run_id=%s graph=%s host=%s pid=%s slurm_job_id=%s array_task_id=%s",
        args.run_id,
        args.graph_name,
        socket.gethostname(),
        os.getpid(),
        os.getenv("SLURM_JOB_ID", "local"),
        os.getenv("SLURM_ARRAY_TASK_ID", "local"),
    )
    manifest = {
        "schema_version": 2,
        "trace_schema_version": 2,
        "status": "running",
        "run_id": args.run_id,
        "graph_name": args.graph_name,
        "graph_mode": graph_mode,
        "dataset": str(args.dataset),
        "num_samples": 0,
        "total_queries": 0,
        "max_hops": args.max_hops,
        "readout_k": args.readout_k,
        "random_seed": args.random_seed if graph_mode == "random-rewiring" else None,
        "embedding_model": str(getattr(config, "EMBEDDING_MODEL", "")),
        "answer_generation_enabled": False,
        "judge_enabled": False,
        "evidence_unit": "source_turn",
        "trace_semantics": {
            "evidence_unit": "source_turn",
            "stage_id_unit": "cell",
            "visited_cell_ids_by_hop": (
                "final_bounded_expansion_cells_grouped_by_first_best_path_hop"
            ),
            "edge_traversal_trace": "all_attempted_local_expansion_edges",
            "first_gold_hit_hop": "minimum_hop_with_gold_cell_intersection",
        },
        "trace_file": str(trace_path),
        "graph_file": str(graph_path),
        "debug_log": str(args.output_dir / "debug.log"),
        "completed_samples": 0,
        "completed_queries": 0,
    }
    write_json(manifest_path, manifest)

    completed_queries = 0
    try:
        mark_stage("dataset_load", path=args.dataset)
        if not args.dataset.is_file():
            raise FileNotFoundError(f"dataset not found: {args.dataset}")
        samples = load_samples(args.dataset, args.num_samples)
        if not samples:
            raise ValueError("dataset selection contains no samples")
        total_queries = sum(len(sample.get("qa") or []) for sample in samples)
        if total_queries == 0:
            raise ValueError("dataset selection contains no questions")
        manifest["num_samples"] = len(samples)
        manifest["total_queries"] = total_queries
        write_json(manifest_path, manifest)
        logger.info(
            "DATASET_READY samples=%s queries=%s roots=%s",
            len(samples),
            total_queries,
            [
                {
                    "sample_idx": index,
                    "sessions": sum(
                        key.startswith("session_") and isinstance(value, list)
                        for key, value in (sample.get("conversation") or {}).items()
                    ),
                    "turns": sum(
                        len(value)
                        for key, value in (sample.get("conversation") or {}).items()
                        if key.startswith("session_") and isinstance(value, list)
                    ),
                    "queries": len(sample.get("qa") or []),
                }
                for index, sample in enumerate(samples)
            ],
        )
        mark_stage("model_init", model=config.EMBEDDING_MODEL)
        model_start = time.perf_counter()
        system = MemRewireSystem(
            clear_db=True,
            enable_parallel_processing=False,
            enable_parallel_retrieval=False,
            graph_mode=graph_mode,
        )
        # Keep the experiment condition explicit even when this runner is
        # used with an older installed MemRewireSystem signature. The graph
        # builder must see the same mode as the run manifest.
        config.THREAD_GRAPH_MODE = graph_mode
        if hasattr(system.vector_store, "graph_mode"):
            system.vector_store.graph_mode = graph_mode
        logger.info(
            "RUNTIME_GRAPH_MODE requested=%s config=%s store=%s runner=%s",
            graph_mode,
            getattr(config, "THREAD_GRAPH_MODE", None),
            getattr(system.vector_store, "graph_mode", "<cell-store>"),
            Path(__file__).resolve(),
        )
        logger.info(
            "STORE_IMPL class=%s module=%s source=%s add_threads_source=%s",
            type(system.vector_store).__name__,
            type(system.vector_store).__module__,
            inspect.getsourcefile(type(system.vector_store)),
            inspect.getsourcefile(type(system.vector_store).add_threads),
        )
        actual_model = Path(system.embedding_model.model_name).expanduser().resolve()
        expected_model = Path(config.EMBEDDING_MODEL).expanduser().resolve()
        if actual_model != expected_model:
            raise RuntimeError(
                f"Embedding fallback detected: expected {expected_model}, got {actual_model}"
            )
        if int(system.embedding_model.dimension) != 1024:
            raise RuntimeError(
                f"Unexpected embedding dimension: {system.embedding_model.dimension}"
            )
        try:
            import torch
            device = str(system.embedding_model.model.device)
            cuda_device = (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else "none"
            )
        except Exception as exc:
            device = "unknown"
            cuda_device = f"unavailable:{type(exc).__name__}"
        logger.info(
            "MODEL_READY path=%s dimension=%s device=%s cuda_device=%s elapsed_seconds=%.2f",
            actual_model,
            system.embedding_model.dimension,
            device,
            cuda_device,
            time.perf_counter() - model_start,
        )
        with trace_path.open("w", encoding="utf-8") as trace_handle, graph_path.open(
            "w", encoding="utf-8"
        ) as graph_handle:
            for sample_idx, sample in enumerate(samples):
                dialogues = convert_to_dialogues(sample)
                mark_stage(
                    "sample_start",
                    sample_idx=sample_idx,
                    sessions=len({dialogue.session_id for dialogue in dialogues}),
                    turns=len(dialogues),
                    queries=len(sample.get("qa") or []),
                )
                mark_stage("bank_clear", sample_idx=sample_idx)
                system.vector_store.clear()
                mark_stage("bank_build_start", sample_idx=sample_idx)
                build_start = time.perf_counter()
                system.add_dialogues(dialogues)
                mark_stage("bank_finalize_start", sample_idx=sample_idx)
                system.finalize()
                logger.info(
                    "BANK_READY sample_idx=%s source_turns=%s cells=%s elapsed_seconds=%.2f",
                    sample_idx,
                    len(dialogues),
                    system.vector_store.count(),
                    time.perf_counter() - build_start,
                )

                mark_stage("graph_snapshot", sample_idx=sample_idx)
                threads = system.vector_store.get_all_threads()
                diagnostics = dict(system.vector_store.last_graph_diagnostics)
                actual_mode = str(diagnostics.get("mode", "")).strip().lower()
                if actual_mode != graph_mode:
                    raise RuntimeError(
                        "Graph mode mismatch: "
                        f"requested={graph_mode!r}, actual={actual_mode!r}, "
                        f"diagnostics={diagnostics}"
                    )
                if diagnostics.get("mode") == "random-rewiring":
                    if not diagnostics.get("degree_sequence_preserved"):
                        raise RuntimeError(
                            f"Random rewiring changed degree sequence: {diagnostics}"
                        )
                    if diagnostics.get("accepted_swaps", 0) == 0:
                        logger.warning(
                            "RANDOM_REWIRING_NO_SWAPS sample_idx=%s diagnostics=%s",
                            sample_idx,
                            diagnostics,
                        )
                logger.info(
                    "GRAPH_READY sample_idx=%s cells=%s edges=%s diagnostics=%s",
                    sample_idx,
                    len(threads),
                    sum(len(thread.links) for thread in threads) // 2,
                    diagnostics,
                )
                append_jsonl(
                    graph_handle,
                    graph_snapshot(
                        threads,
                        args.graph_name,
                        sample_idx,
                        args.run_id,
                        diagnostics,
                    ),
                )
                source_to_cell = {
                    str(source_id): thread.thread_id
                    for thread in threads
                    for node in thread.nodes
                    for source_id in node.source_turn_ids
                }

                for question_idx, qa in enumerate(sample.get("qa") or []):
                    question = str(qa.get("question") or "")
                    gold_turn_ids = normalize_evidence_ids(qa.get("evidence"))
                    query_id = f"locomo10:{sample_idx:02d}:qa:{question_idx:03d}"
                    mark_stage(
                        "query_retrieval_start",
                        query_id=query_id,
                        query_index=f"{completed_queries + 1}/{total_queries}",
                        category=category_name(qa.get("category")),
                        gold_turns=len(gold_turn_ids),
                    )
                    started = time.perf_counter()
                    workspace = system.hybrid_retriever.retrieve_workspace(question)
                    latency_ms = (time.perf_counter() - started) * 1000.0

                    visited_cell_ids_by_hop = [
                        {
                            "hop": int(group.get("hop", 0)),
                            "cell_ids": [
                                str(cell_id)
                                for cell_id in (group.get("cell_ids") or [])
                            ],
                        }
                        for group in (
                            getattr(system.vector_store, "last_expansion_by_hop", [])
                            or []
                        )
                    ]
                    if not visited_cell_ids_by_hop:
                        visited_cell_ids_by_hop = [
                            {"hop": hop, "cell_ids": []}
                            for hop in range(args.max_hops + 1)
                        ]
                    edge_traversal_trace = [
                        dict(edge)
                        for edge in (
                            getattr(system.vector_store, "last_expansion_trace", [])
                            or []
                        )
                    ]

                    routed_turn_ids = sorted(
                        {str(value) for value in workspace.routed_source_turn_ids}
                    )
                    visited_turn_ids = sorted(
                        {str(value) for value in workspace.expanded_source_turn_ids}
                    )
                    selected_turn_ids = list(
                        dict.fromkeys(
                            str(source_id)
                            for cell in workspace.assembly_cells
                            for source_id in cell.source_turn_ids
                        )
                    )
                    gold_turn_set = set(gold_turn_ids)
                    routed_recall = (
                        len(gold_turn_set & set(routed_turn_ids)) / len(gold_turn_set)
                        if gold_turn_set else None
                    )
                    visited_recall = (
                        len(gold_turn_set & set(visited_turn_ids)) / len(gold_turn_set)
                        if gold_turn_set else None
                    )
                    selected_recall = (
                        len(gold_turn_set & set(selected_turn_ids)) / len(gold_turn_set)
                        if gold_turn_set else None
                    )
                    gold_cell_ids = sorted(
                        {
                            source_to_cell[value]
                            for value in gold_turn_ids
                            if value in source_to_cell
                        }
                    )
                    first_gold_hit_hop = next(
                        (
                            group["hop"]
                            for group in visited_cell_ids_by_hop
                            if set(group["cell_ids"]) & set(gold_cell_ids)
                        ),
                        None,
                    )
                    route_stage_ids = list(dict.fromkeys(
                        str(cell_id) for cell_id in workspace.routed_thread_ids
                    ))
                    expansion_stage_ids = list(dict.fromkeys(
                        str(cell_id) for cell_id in workspace.expanded_thread_ids
                    ))
                    candidate_stage_ids = list(dict.fromkeys(
                        str(cell_id)
                        for cell_id in getattr(workspace, "candidate_thread_ids", [])
                    ))
                    selected_stage_ids = list(dict.fromkeys(
                        str(cell.thread_id) for cell in workspace.assembly_cells
                    ))
                    logger.info(
                        "QUERY_TRACE query_id=%s hops=%s attempted_edges=%s "
                        "first_gold_hit_hop=%s route_stage=%s expansion_stage=%s "
                        "candidate_stage=%s selected_stage=%s",
                        query_id,
                        [
                            len(group["cell_ids"])
                            for group in visited_cell_ids_by_hop
                        ],
                        len(edge_traversal_trace),
                        first_gold_hit_hop,
                        len(route_stage_ids),
                        len(expansion_stage_ids),
                        len(candidate_stage_ids),
                        len(selected_stage_ids),
                    )
                    record = {
                        "schema_version": 2,
                        "trace_schema_version": 2,
                        "run_id": args.run_id,
                        "sample_idx": sample_idx,
                        "sample_id": str(sample.get("sample_id", sample_idx)),
                        "question_idx": question_idx,
                        "query_id": query_id,
                        "question": question,
                        "question_type": category_name(qa.get("category")),
                        "question_category": qa.get("category"),
                        "graph_name": args.graph_name,
                        "max_hops": args.max_hops,
                        "readout_k": args.readout_k,
                        "gold_turn_ids": gold_turn_ids,
                        "gold_cell_ids": gold_cell_ids,
                        "unmapped_gold_turn_ids": [
                            value for value in gold_turn_ids if value not in source_to_cell
                        ],
                        "seed_cell_ids": sorted(set(workspace.routed_thread_ids)),
                        "visited_cell_ids": sorted(set(workspace.expanded_thread_ids)),
                        "candidate_cell_ids": candidate_stage_ids,
                        "candidate_turn_ids": visited_turn_ids,
                        "selected_cell_ids": sorted(
                            {
                                str(cell.thread_id)
                                for cell in workspace.assembly_cells
                            }
                        ),
                        "routed_turn_ids": routed_turn_ids,
                        "visited_turn_ids": visited_turn_ids,
                        "selected_turn_ids": selected_turn_ids,
                        "visited_cell_ids_by_hop": visited_cell_ids_by_hop,
                        "edge_traversal_trace": edge_traversal_trace,
                        "first_gold_hit_hop": first_gold_hit_hop,
                        "route_stage_ids": route_stage_ids,
                        "expansion_stage_ids": expansion_stage_ids,
                        "candidate_stage_ids": candidate_stage_ids,
                        "selected_stage_ids": selected_stage_ids,
                        "candidate_node_ids": [
                            str(node_id)
                            for node_id in getattr(workspace, "candidate_node_ids", [])
                        ],
                        "evidence_recall": visited_recall,
                        "complete_evidence_recovery": (
                            gold_turn_set.issubset(set(visited_turn_ids))
                            if gold_turn_set else None
                        ),
                        "routed_evidence_recall": routed_recall,
                        "selected_context_evidence_recall": selected_recall,
                        "answer_correct": None,
                        "access_tokens": None,
                        "latency_ms": latency_ms,
                        "graph_diagnostics": diagnostics,
                    }
                    append_jsonl(trace_handle, record)
                    completed_queries += 1
                    logger.info(
                        "QUERY_DONE graph=%s queries=%s/%s query_id=%s "
                        "category=%s recall=%s routed_recall=%s selected_recall=%s "
                        "routed_cells=%s visited_cells=%s selected_cells=%s "
                        "latency_ms=%.1f trace_bytes=%s",
                        args.graph_name,
                        completed_queries,
                        total_queries,
                        query_id,
                        record["question_type"],
                        visited_recall,
                        routed_recall,
                        selected_recall,
                        len(record["seed_cell_ids"]),
                        len(record["visited_cell_ids"]),
                        len(record["selected_cell_ids"]),
                        latency_ms,
                        trace_path.stat().st_size,
                    )

                    manifest["completed_queries"] = completed_queries
                    write_json(manifest_path, manifest)

                manifest["completed_samples"] = sample_idx + 1
                write_json(manifest_path, manifest)

        manifest["status"] = "completed"
        manifest["completed_queries"] = completed_queries
        manifest["completed_at"] = time.time()
        write_json(manifest_path, manifest)
        completion_path.touch()
        mark_stage("completed", queries=completed_queries, samples=len(samples))
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["completed_queries"] = completed_queries
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, manifest)
        logger.exception(
            "RUN_FAILED stage=%s completed_queries=%s error=%s",
            current_stage["name"],
            completed_queries,
            manifest["error"],
        )
        raise


if __name__ == "__main__":
    main()
