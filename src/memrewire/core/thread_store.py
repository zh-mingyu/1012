"""Persistent cell index and local traversal over fixed graph links."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import json
import logging
import math
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # LanceDB is used in the full experiment environment.
    import lancedb
    import pyarrow as pa
    _HAS_LANCEDB = True
except ImportError:  # The lightweight local test runtime has no LanceDB.
    lancedb = None
    pa = None
    _HAS_LANCEDB = False

from memwire.core.models.memory_entry import (
    InteractionThread,
    MemoryEntry,
    ThreadLink,
    ThreadNode,
)
from memwire.core.settings import settings as config
from memwire.core.utils.embedding import EmbeddingModel


@dataclass
class ThreadRoute:
    thread: InteractionThread
    score: float
    rank: int


@dataclass
class ThreadCandidate:
    thread: InteractionThread
    route_score: float
    hop: int
    path: List[str]


class ThreadStore:
    """Store routing addresses globally and expose fixed local payload links."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        table_name: Optional[str] = None,
        graph_mode: Optional[str] = None,
    ) -> None:
        self.db_path = db_path or getattr(config, "THREAD_LANCEDB_PATH", "./thread_lancedb_data")
        self.table_name = table_name or getattr(config, "THREAD_TABLE_NAME", "interaction_threads")
        configured_graph_mode = graph_mode
        if configured_graph_mode is None:
            configured_graph_mode = getattr(config, "THREAD_GRAPH_MODE", "operator-aware")
        self.graph_mode = str(configured_graph_mode).strip().lower()
        self.head_table_name = f"{self.table_name}_heads"
        self.payload_table_name = f"{self.table_name}_payloads"
        self.embedding_model = embedding_model or EmbeddingModel()
        self._rows: List[Dict[str, Any]] = []
        self._payload_rows: List[Dict[str, Any]] = []
        self._payload_cache: Dict[str, Dict[str, Any]] = {}
        self._thread_cache: Dict[str, InteractionThread] = {}
        self._head_cache: Dict[str, InteractionThread] = {}
        # A local receptive-field expansion touches many small LanceDB rows.
        # Build these lightweight metadata maps lazily once per sample so the
        # query path does not issue one point lookup per traversed edge.
        self._payload_index: Optional[Dict[str, Dict[str, Any]]] = None
        self._head_index: Optional[Dict[str, Dict[str, Any]]] = None
        self._payload_index_load_failed = False
        self._head_index_load_failed = False
        self._metadata_index_lock = threading.Lock()
        self.last_route_comparisons = 0
        self.last_expansion_edges = 0
        self.last_expansion_trace: List[Dict[str, Any]] = []
        self.last_expansion_by_hop: List[Dict[str, Any]] = []
        # Offline topology diagnostics.  These values describe construction
        # of the persistent graph; they are never query-conditioned state.
        self.last_graph_diagnostics: Dict[str, Any] = {
            "mode": self.graph_mode,
            "candidate_edges": 0,
            "selected_optional_edges": 0,
            "mandatory_edges": 0,
            "objective": 0.0,
        }
        self._use_lancedb = bool(_HAS_LANCEDB)
        if self._use_lancedb:
            self._is_cloud_storage = self.db_path.startswith(("gs://", "s3://", "az://"))
            if self._is_cloud_storage:
                self.db = lancedb.connect(self.db_path)
            else:
                os.makedirs(self.db_path, exist_ok=True)
                self.db = lancedb.connect(self.db_path)
            self._init_tables()
        else:
            os.makedirs(self.db_path, exist_ok=True)
            self._fallback_path = os.path.join(self.db_path, f"{self.table_name}.json")
            self._load_fallback()

    def _init_tables(self) -> None:
        head_schema = pa.schema(
            [
                pa.field("thread_id", pa.string()),
                pa.field("head", pa.string()),
                pa.field("session_id", pa.string()),
                pa.field("ordinal", pa.int64()),
                pa.field("start_timestamp", pa.string()),
                pa.field("end_timestamp", pa.string()),
                pa.field("links_json", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension)),
            ]
        )
        payload_schema = pa.schema(
            [
                pa.field("thread_id", pa.string()),
                pa.field("nodes_json", pa.string()),
            ]
        )
        table_names = set(self.db.table_names())
        if self.head_table_name not in table_names:
            self.table = self.db.create_table(self.head_table_name, schema=head_schema)
        else:
            self.table = self.db.open_table(self.head_table_name)
        if self.payload_table_name not in table_names:
            self.payload_table = self.db.create_table(
                self.payload_table_name,
                schema=payload_schema,
            )
        else:
            self.payload_table = self.db.open_table(self.payload_table_name)

    def _load_fallback(self) -> None:
        try:
            with open(self._fallback_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self._rows = payload if isinstance(payload, list) else []
            self._payload_rows = []
            self._payload_cache = {}
            self._thread_cache = {}
            self._head_cache = {}
            self._payload_index = None
            self._head_index = None
            self._payload_index_load_failed = False
            self._head_index_load_failed = False
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self._rows = []

    def _save_fallback(self) -> None:
        temp_path = self._fallback_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(self._rows, handle, ensure_ascii=True)
        os.replace(temp_path, self._fallback_path)

    def count(self) -> int:
        return self.table.count_rows() if self._use_lancedb else len(self._rows)

    def clear(self) -> None:
        if self._use_lancedb:
            for name in (self.head_table_name, self.payload_table_name):
                if name in self.db.table_names():
                    self.db.drop_table(name)
            self._init_tables()
            self._payload_cache = {}
            self._thread_cache = {}
            self._head_cache = {}
            self._payload_index = None
            self._head_index = None
            self._payload_index_load_failed = False
            self._head_index_load_failed = False
        else:
            self._rows = []
            self._payload_cache = {}
            self._thread_cache = {}
            self._head_cache = {}
            self._payload_index = None
            self._head_index = None
            self._payload_index_load_failed = False
            self._head_index_load_failed = False
            self._save_fallback()

    def add_threads(self, threads: Sequence[InteractionThread]) -> None:
        if not threads:
            return
        graph_started = time.perf_counter()
        logger = logging.getLogger("paired_graph.graph")
        existing = {thread.thread_id: thread for thread in self.get_all_threads()}
        incoming = [thread.copy(deep=True) for thread in threads]
        logger.info(
            "GRAPH_BUILD_START mode=%s incoming_cells=%s existing_cells=%s",
            getattr(config, "THREAD_GRAPH_MODE", "operator-aware"),
            len(incoming),
            len(existing),
        )
        heads = [thread.routing_text for thread in incoming]
        head_vectors = self.embedding_model.encode_documents(heads)
        # Source selection is defined over the original turns, so each node
        # embedding must represent the node text itself.  The cell head is
        # embedded separately and is used only for coarse cell routing.
        raw_node_texts = [
            node.text
            for thread in incoming
            for node in thread.nodes
        ]
        node_vectors = []
        if raw_node_texts:
            node_vectors = [
                self._as_list(vector)
                for vector in self.embedding_model.encode_documents(raw_node_texts)
            ]
        node_index = 0
        for thread, head_vector in zip(incoming, head_vectors):
            thread.embedding = [float(value) for value in self._as_list(head_vector)]
            for node in thread.nodes:
                if len(node_vectors):
                    node.embedding = [float(value) for value in node_vectors[node_index]]
                node_index += 1
            existing[thread.thread_id] = thread
        materialized = sorted(
            existing.values(),
            key=lambda thread: (thread.ordinal, thread.thread_id),
        )
        # Links are materialized once from source order and static locator
        # features. The operator-aware construction objective is query-free;
        # the query path sees only the resulting fixed graph.
        # The experiment runner passes this explicitly at construction time.
        # Do not re-read the mutable process-wide settings here: that allowed
        # a condition labelled random-rewiring to silently build operator-aware
        # links when another import or environment layer changed the setting.
        graph_mode = self.graph_mode
        normalized_graph_mode = graph_mode.replace("_", "-").strip().lower()
        logger.info(
            "GRAPH_MODE_RESOLVED self=%s normalized=%s config=%s class=%s module=%s",
            graph_mode,
            normalized_graph_mode,
            getattr(config, "THREAD_GRAPH_MODE", None),
            type(self).__name__,
            type(self).__module__,
        )
        if normalized_graph_mode in {"random", "random-rewire", "random-rewiring"}:
            logger.info("GRAPH_BRANCH selected=random-rewiring")
            self.last_graph_diagnostics = self._random_rewire_links(
                materialized,
                max_neighbors=int(getattr(config, "THREAD_GRAPH_NEIGHBORS", 3)),
                min_similarity=float(getattr(config, "THREAD_GRAPH_MIN_SIM", 0.25)),
                seed=int(getattr(config, "THREAD_GRAPH_RANDOM_SEED", 2026)),
                swaps_per_edge=int(
                    getattr(config, "THREAD_GRAPH_RANDOM_SWAPS_PER_EDGE", 10)
                ),
            )
        elif normalized_graph_mode in {"initial", "legacy", "cosine", "knn"}:
            logger.info("GRAPH_BRANCH selected=initial")
            self._build_thread_links(
                materialized,
                max_neighbors=int(getattr(config, "THREAD_GRAPH_NEIGHBORS", 3)),
                min_similarity=float(getattr(config, "THREAD_GRAPH_MIN_SIM", 0.25)),
            )
            self.last_graph_diagnostics = {
                "mode": "initial",
                "candidate_edges": 0,
                "selected_optional_edges": sum(len(thread.links) for thread in materialized) // 2,
                "mandatory_edges": 0,
                "objective": 0.0,
            }
        else:
            logger.info("GRAPH_BRANCH selected=operator-aware")
            graph_compile_started = time.perf_counter()
            self.last_graph_diagnostics = self._compile_operator_links(
                materialized,
                max_neighbors=int(getattr(config, "THREAD_GRAPH_NEIGHBORS", 3)),
                min_similarity=float(getattr(config, "THREAD_GRAPH_MIN_SIM", 0.25)),
                candidate_multiplier=int(getattr(config, "THREAD_OPERATOR_CANDIDATE_MULTIPLIER", 3)),
                sketch_dim=int(getattr(config, "THREAD_OPERATOR_SKETCH_DIM", 6)),
                objective_scale=float(getattr(config, "THREAD_OPERATOR_OBJECTIVE_SCALE", 1.0)),
                swap_search_width=int(getattr(config, "THREAD_OPERATOR_SWAP_SEARCH_WIDTH", 256)),
                max_swaps=int(getattr(config, "THREAD_OPERATOR_MAX_SWAPS", 12)),
            )
            logger.info(
                "GRAPH_COMPILE_DONE mode=operator-aware cells=%s elapsed_seconds=%.2f diagnostics=%s",
                len(materialized),
                time.perf_counter() - graph_compile_started,
                self.last_graph_diagnostics,
            )
        self._thread_cache = {
            thread.thread_id: thread.copy(deep=True)
            for thread in materialized
        }
        self._head_cache = {
            thread.thread_id: InteractionThread(
                thread_id=thread.thread_id,
                head=thread.head,
                session_id=thread.session_id,
                ordinal=thread.ordinal,
                start_timestamp=thread.start_timestamp,
                end_timestamp=thread.end_timestamp,
                links=[link.copy(deep=True) for link in thread.links],
                embedding=list(thread.embedding),
            )
            for thread in materialized
        }
        rows = [self._thread_to_head_row(thread) for thread in materialized]
        payload_rows = [self._thread_to_payload_row(thread) for thread in materialized]
        if self._use_lancedb:
            for name in (self.head_table_name, self.payload_table_name):
                if name in self.db.table_names():
                    self.db.drop_table(name)
            self._init_tables()
            self._payload_cache = {}
            self.table.add(rows)
            self.payload_table.add(payload_rows)
        else:
            # The dependency-free fallback keeps one JSON file for portability;
            # the production LanceDB path physically separates heads and nodes.
            self._rows = [self._thread_to_row(thread) for thread in materialized]
            self._save_fallback()
        # Tables are rebuilt after every LoCoMo root. Invalidate the lazy maps
        # so no payload/head from the previous root can leak into expansion.
        self._payload_index = None
        self._head_index = None
        self._payload_index_load_failed = False
        self._head_index_load_failed = False
        logger.info(
            "GRAPH_BUILD_DONE mode=%s cells=%s edges=%s elapsed_seconds=%.2f",
            self.last_graph_diagnostics.get("mode", graph_mode),
            len(materialized),
            sum(len(thread.links) for thread in materialized) // 2,
            time.perf_counter() - graph_started,
        )

    def route(
        self,
        address: str,
        top_k: Optional[int] = 4,
    ) -> Tuple[List[ThreadRoute], List[float]]:
        """Score cell locator texts and optionally keep the highest scores.

        The canonical paper path calls this method with ``top_k=None`` and
        applies the mean-plus-standard-deviation seed rule over all cells.
        A finite ``top_k`` remains available for compatibility and ablations.
        """
        if self.count() == 0:
            return [], []
        query_vector = self.embedding_model.encode_single(address, is_query=True)
        query_values = self._as_list(query_vector)
        if self._use_lancedb:
            query = self.table.search(query_values)
            raw = (
                query.limit(max(1, self.count())).to_list()
                if top_k is None
                else query.limit(max(1, int(top_k))).to_list()
            )
            scored = [
                (row, self._cosine(query_values, row.get("vector") or []))
                for row in raw
            ]
            scored.sort(key=lambda item: item[1], reverse=True)
        else:
            raw = []
            for row in self._rows:
                score = self._cosine(query_values, row.get("vector") or [])
                raw.append((row, score))
            scored = raw
            scored.sort(key=lambda item: item[1], reverse=True)
            if top_k is not None:
                scored = scored[: max(1, int(top_k))]
        self.last_route_comparisons = self.count()
        return [
            ThreadRoute(self._hydrate_thread(row), float(score), rank)
            for rank, (row, score) in enumerate(scored, 1)
        ], [float(value) for value in query_values]

    def route_seeds(self, address: str) -> Tuple[List[ThreadRoute], List[float]]:
        """Return cells above the paper's mean-plus-standard-deviation cutoff."""
        routes, query_vector = self.route(address, top_k=None)
        if not routes:
            return [], query_vector
        scores = [route.score for route in routes]
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        threshold = mean + math.sqrt(max(0.0, variance))
        seeds = [route for route in routes if route.score >= threshold]
        if not seeds:
            seeds = [routes[0]]
        return seeds, query_vector

    @staticmethod
    def _as_list(value: Any) -> List[float]:
        """Convert numpy/torch/list embedding outputs to plain floats."""
        if hasattr(value, "tolist"):
            value = value.tolist()
        return list(value)

    @staticmethod
    def _node_routing_text(thread: InteractionThread, node: ThreadNode) -> str:
        timestamp = f" [{node.timestamp}]" if node.timestamp else ""
        return f"{thread.head}{timestamp}\n{node.text}"[:4000]

    @staticmethod
    def _blend_vectors(raw: Sequence[float], contextual: Sequence[float]) -> List[float]:
        if len(raw) != len(contextual):
            return [float(value) for value in raw]
        values = [0.6 * float(left) + 0.4 * float(right) for left, right in zip(raw, contextual)]
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values] if norm else values

    def get_all_threads(self) -> List[InteractionThread]:
        if not self._use_lancedb:
            if not self._thread_cache and self._rows:
                self._thread_cache = {
                    thread.thread_id: thread
                    for thread in (self._row_to_thread(row) for row in self._rows)
                }
            return [thread.copy(deep=True) for thread in self._thread_cache.values()]
        if self._thread_cache:
            return [thread.copy(deep=True) for thread in self._thread_cache.values()]
        head_rows = self.table.to_arrow().to_pylist()
        payload_rows = self.payload_table.to_arrow().to_pylist()
        payload_by_id = {str(row["thread_id"]): row for row in payload_rows}
        threads = [
            self._hydrate_thread(row, payload_by_id.get(str(row["thread_id"])))
            for row in head_rows
        ]
        self._thread_cache = {thread.thread_id: thread for thread in threads}
        self._head_cache = {
            thread.thread_id: self._row_to_head_thread(self._thread_to_head_row(thread))
            for thread in threads
        }
        return [thread.copy(deep=True) for thread in threads]

    def get_all_entries(self) -> List[MemoryEntry]:
        entries: List[MemoryEntry] = []
        for thread in self.get_all_threads():
            for node in thread.nodes:
                entries.append(MemoryEntry(
                    entry_id=node.node_id,
                    lossless_restatement=node.text,
                    timestamp=node.timestamp,
                    source_turn_ids=node.source_turn_ids,
                    thread_id=thread.thread_id,
                ))
        return entries

    def expand(
        self,
        routes: Sequence[ThreadRoute],
        max_hops: int = 2,
        max_threads: Optional[int] = None,
    ) -> List[ThreadCandidate]:
        """Traverse routed links with best-first score propagation.

        FIFO traversal made the result depend on route order: a weak early
        neighbor could hide a stronger path before it was inspected. The
        receptive-field traversal remains local and bounded, while candidates
        are admitted by propagated route score.
        """
        self.last_expansion_edges = 0
        self.last_expansion_trace = []
        self.last_expansion_by_hop = []
        if not routes:
            return []
        # Seed the cache with routed heads, then fetch only the head metadata
        # named by a traversed edge. This avoids a first-query scan of the
        # complete head table; payloads are hydrated only for reached threads.
        headers = self._head_cache
        for route in routes:
            headers.setdefault(
                route.thread.thread_id,
                self._row_to_head_thread(self._thread_to_head_row(route.thread)),
            )
        frontier: List[Tuple[float, int, int, str, int, ThreadCandidate]] = []
        push_order = 0
        # The method defines one propagation score for every (cell, hop)
        # state. Keeping only one score per cell can discard a lower-scoring
        # shorter path that is still able to reach another cell within H hops.
        best_by_hop: Dict[Tuple[str, int], ThreadCandidate] = {}
        discovered_by_hop: Dict[int, set[str]] = {}
        for route in routes:
            candidate = ThreadCandidate(
                self._hydrate_thread_from_header(route.thread),
                route.score,
                0,
                [route.thread.thread_id],
            )
            state_key = (route.thread.thread_id, 0)
            previous = best_by_hop.get(state_key)
            if previous is None or candidate.route_score > previous.route_score:
                best_by_hop[state_key] = candidate
                discovered_by_hop.setdefault(0, set()).add(route.thread.thread_id)
                heapq.heappush(
                    frontier,
                    (-candidate.route_score, candidate.hop, route.rank, candidate.thread.thread_id, push_order, candidate),
                )
                push_order += 1
        if max_threads is not None:
            max_threads = max(1, int(max_threads))
        while frontier:
            _neg_score, _hop, _rank, _thread_id, _push_order, current = heapq.heappop(frontier)
            state_key = (current.thread.thread_id, current.hop)
            incumbent = best_by_hop.get(state_key)
            if incumbent is None or incumbent.path != current.path or incumbent.route_score > current.route_score + 1e-12:
                continue
            if current.hop >= max(0, int(max_hops)):
                continue
            for link in sorted(
                current.thread.links,
                key=lambda item: (-float(item.weight), item.target_thread_id),
            ):
                self.last_expansion_edges += 1
                next_hop = current.hop + 1
                target_header = headers.get(link.target_thread_id)
                if target_header is None:
                    target_header = self._load_head_thread(link.target_thread_id)
                    if target_header is not None:
                        headers[target_header.thread_id] = target_header
                if target_header is None:
                    self.last_expansion_trace.append(
                        {
                            "from_cell_id": str(current.thread.thread_id),
                            "to_cell_id": str(link.target_thread_id),
                            "from_hop": int(current.hop),
                            "to_hop": int(next_hop),
                            "relation": str(link.relation),
                            "weight": float(link.weight),
                            "next_score": None,
                            "accepted": False,
                            "reason": "missing_target",
                        }
                    )
                    continue
                target = self._hydrate_thread_from_header(target_header)
                if target is None:
                    self.last_expansion_trace.append(
                        {
                            "from_cell_id": str(current.thread.thread_id),
                            "to_cell_id": str(target_header.thread_id),
                            "from_hop": int(current.hop),
                            "to_hop": int(next_hop),
                            "relation": str(link.relation),
                            "weight": float(link.weight),
                            "next_score": None,
                            "accepted": False,
                            "reason": "missing_target",
                        }
                    )
                    continue
                # The paper uses the pathwise average recurrence
                # r_{t+1}=(r_t+w(C_t,C_{t+1}))/2.
                next_score = 0.5 * (
                    current.route_score + float(link.weight)
                )
                next_key = (target.thread_id, next_hop)
                previous = best_by_hop.get(next_key)
                if previous is not None and previous.route_score >= next_score:
                    self.last_expansion_trace.append(
                        {
                            "from_cell_id": str(current.thread.thread_id),
                            "to_cell_id": str(target_header.thread_id),
                            "from_hop": int(current.hop),
                            "to_hop": int(next_hop),
                            "relation": str(link.relation),
                            "weight": float(link.weight),
                            "next_score": float(next_score),
                            "accepted": False,
                            "reason": "score_not_better",
                        }
                    )
                    continue
                candidate = ThreadCandidate(target, next_score, next_hop, current.path + [target.thread_id])
                best_by_hop[next_key] = candidate
                discovered_by_hop.setdefault(next_hop, set()).add(target.thread_id)
                self.last_expansion_trace.append(
                    {
                        "from_cell_id": str(current.thread.thread_id),
                        "to_cell_id": str(target.thread_id),
                        "from_hop": int(current.hop),
                        "to_hop": int(next_hop),
                        "relation": str(link.relation),
                        "weight": float(link.weight),
                        "next_score": float(next_score),
                        "accepted": True,
                        "reason": "accepted",
                    }
                )
                heapq.heappush(
                    frontier,
                    (-candidate.route_score, candidate.hop, 999, candidate.thread.thread_id, push_order, candidate),
                )
                push_order += 1
        route_order = {route.thread.thread_id: route.rank for route in routes}
        best: Dict[str, ThreadCandidate] = {}
        for candidate in best_by_hop.values():
            previous = best.get(candidate.thread.thread_id)
            if previous is None or (
                candidate.route_score > previous.route_score + 1e-12
                or (
                    abs(candidate.route_score - previous.route_score) <= 1e-12
                    and candidate.hop < previous.hop
                )
            ):
                best[candidate.thread.thread_id] = candidate
        # A local expansion may add context, but it must never discard a
        # routed seed. Preserve all seeds first, then fill the receptive field
        # with the strongest propagated neighbors.
        routed = [
            item
            for item in best.values()
            if item.thread.thread_id in route_order
        ]
        routed.sort(key=lambda item: route_order[item.thread.thread_id])
        routed_ids = {item.thread.thread_id for item in routed}
        neighbors = sorted(
            (
                item
                for item in best.values()
                if item.thread.thread_id not in routed_ids
            ),
            key=lambda item: (
                -item.route_score,
                item.hop,
                item.thread.ordinal,
                item.thread.thread_id,
            ),
        )
        bounded = routed + neighbors
        if max_threads is not None:
            bounded = bounded[:max_threads]
        hop_limit = max(0, int(max_hops))
        self.last_expansion_by_hop = [
            {
                "hop": hop,
                "cell_ids": sorted(discovered_by_hop.get(hop, set())),
            }
            for hop in range(hop_limit + 1)
        ]
        return bounded

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
        right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    @staticmethod
    def _build_thread_links(
        threads: List[InteractionThread],
        max_neighbors: int = 3,
        min_similarity: float = 0.25,
    ) -> None:
        """Build the paper's initial cell graph.

        Semantic edges use the global off-diagonal mean-plus-standard-
        deviation threshold. ``max_neighbors`` and ``min_similarity`` remain
        accepted for compatibility with older callers but are not used by the
        canonical construction.
        """
        logger = logging.getLogger("paired_graph.graph")
        pairwise_started = time.perf_counter()
        logger.info("GRAPH_BASELINE_START cells=%s", len(threads))
        for thread in threads:
            thread.links = []
        similarities: Dict[Tuple[str, str], float] = {}
        def add_edge(left: InteractionThread, right: InteractionThread, relation: str, weight: float) -> None:
            # The same pair can receive a chronology edge and a semantic edge;
            # retain the stronger interpretation once.
            for existing in left.links:
                if existing.target_thread_id == right.thread_id:
                    existing.weight = max(existing.weight, weight)
                    if existing.relation != "chronology-next" and relation == "chronology-next":
                        existing.relation = relation
                    return
            left.links.append(ThreadLink(
                target_thread_id=right.thread_id,
                relation=relation,
                weight=float(weight),
            ))

        chronology_pairs: set[Tuple[str, str]] = set()
        ordered = sorted(threads, key=lambda item: (item.ordinal, item.thread_id))
        for left, right in zip(ordered, ordered[1:]):
            chronology_pairs.add((left.thread_id, right.thread_id))
            chronology_pairs.add((right.thread_id, left.thread_id))

        report_every = max(1, len(threads) // 10)
        for index, left in enumerate(threads):
            if index % report_every == 0:
                logger.info(
                    "GRAPH_BASELINE_PROGRESS rows=%s/%s elapsed_seconds=%.1f",
                    index,
                    len(threads),
                    time.perf_counter() - pairwise_started,
                )
            for right in threads[index + 1:]:
                adjacent = (left.thread_id, right.thread_id) in chronology_pairs
                similarity = ThreadStore._cosine(left.embedding, right.embedding)
                similarities[tuple(sorted((left.thread_id, right.thread_id)))] = similarity
                if adjacent:
                    add_edge(left, right, "chronology-next", 1.0)
                    add_edge(right, left, "chronology-next", 1.0)

        values = list(similarities.values())
        mean = sum(values) / max(1, len(values))
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
        threshold = mean + math.sqrt(max(0.0, variance))
        by_id = {thread.thread_id: thread for thread in threads}
        for (left_id, right_id), similarity in similarities.items():
            if similarity < threshold:
                continue
            left, right = by_id[left_id], by_id[right_id]
            weight = (similarity + 1.0) / 2.0
            add_edge(left, right, "semantic", weight)
            add_edge(right, left, "semantic", weight)
        logger.info(
            "GRAPH_BASELINE_DONE cells=%s edges=%s elapsed_seconds=%.1f",
            len(threads),
            sum(len(thread.links) for thread in threads) // 2,
            time.perf_counter() - pairwise_started,
        )

    @staticmethod
    def _random_rewire_links(
        threads: List[InteractionThread],
        max_neighbors: int = 3,
        min_similarity: float = 0.25,
        seed: int = 2026,
        swaps_per_edge: int = 10,
    ) -> Dict[str, Any]:
        """Randomize optional semantic edges with chronology and degrees fixed."""
        logger = logging.getLogger("paired_graph.graph")
        rewiring_started = time.perf_counter()
        ThreadStore._build_thread_links(
            threads,
            max_neighbors=max_neighbors,
            min_similarity=min_similarity,
        )

        def key(left: str, right: str) -> Tuple[str, str]:
            return (left, right) if left < right else (right, left)

        chronology: set[Tuple[str, str]] = set()
        optional: set[Tuple[str, str]] = set()
        for thread in threads:
            for link in thread.links:
                pair = key(thread.thread_id, link.target_thread_id)
                if link.relation == "chronology-next":
                    chronology.add(pair)
                else:
                    optional.add(pair)

        edges = sorted(optional)
        edge_set = set(edges)
        degrees = {thread.thread_id: 0 for thread in threads}
        for left, right in edges:
            degrees[left] += 1
            degrees[right] += 1

        rng = random.Random(int(seed))
        requested_swaps = max(0, int(swaps_per_edge)) * len(edges)
        attempts = 0
        accepted = 0
        max_attempts = max(1000, requested_swaps * 50)
        last_progress_log = time.perf_counter()
        logger.info(
            "GRAPH_RANDOM_REWIRE_START optional_edges=%s requested_swaps=%s max_attempts=%s seed=%s",
            len(edges),
            requested_swaps,
            max_attempts,
            seed,
        )
        while len(edges) >= 2 and accepted < requested_swaps and attempts < max_attempts:
            attempts += 1
            if (
                attempts % 10000 == 0
                or time.perf_counter() - last_progress_log >= 30.0
            ):
                logger.info(
                    "GRAPH_RANDOM_REWIRE_PROGRESS attempts=%s/%s accepted=%s/%s elapsed_seconds=%.1f",
                    attempts,
                    max_attempts,
                    accepted,
                    requested_swaps,
                    time.perf_counter() - rewiring_started,
                )
                last_progress_log = time.perf_counter()
            first_index, second_index = rng.sample(range(len(edges)), 2)
            left_a, left_b = edges[first_index]
            right_a, right_b = edges[second_index]
            if len({left_a, left_b, right_a, right_b}) != 4:
                continue

            if rng.getrandbits(1):
                replacements = (key(left_a, right_a), key(left_b, right_b))
            else:
                replacements = (key(left_a, right_b), key(left_b, right_a))
            old_edges = {edges[first_index], edges[second_index]}
            if (
                replacements[0] == replacements[1]
                or any(pair in chronology for pair in replacements)
                or any(pair in edge_set and pair not in old_edges for pair in replacements)
            ):
                continue

            edge_set.difference_update(old_edges)
            edge_set.update(replacements)
            edges[first_index], edges[second_index] = replacements
            accepted += 1

        thread_by_id = {thread.thread_id: thread for thread in threads}
        for thread in threads:
            thread.links = []
        for left, right in sorted(chronology):
            thread_by_id[left].links.append(ThreadLink(
                target_thread_id=right,
                relation="chronology-next",
                weight=1.0,
            ))
            thread_by_id[right].links.append(ThreadLink(
                target_thread_id=left,
                relation="chronology-next",
                weight=1.0,
            ))
        for left, right in sorted(edge_set):
            similarity = ThreadStore._cosine(
                thread_by_id[left].embedding,
                thread_by_id[right].embedding,
            )
            weight = (similarity + 1.0) / 2.0
            thread_by_id[left].links.append(ThreadLink(
                target_thread_id=right,
                relation="semantic",
                weight=weight,
            ))
            thread_by_id[right].links.append(ThreadLink(
                target_thread_id=left,
                relation="semantic",
                weight=weight,
            ))

        final_degrees = {thread.thread_id: 0 for thread in threads}
        for left, right in edge_set:
            final_degrees[left] += 1
            final_degrees[right] += 1
        diagnostics = {
            "mode": "random-rewiring",
            "candidate_edges": 0,
            "selected_edges": len(chronology) + len(edge_set),
            "selected_optional_edges": len(edge_set),
            "mandatory_edges": len(chronology),
            "baseline_edge_budget": len(chronology) + len(edge_set),
            "random_seed": int(seed),
            "requested_swaps": requested_swaps,
            "accepted_swaps": accepted,
            "swap_attempts": attempts,
            "degree_sequence_preserved": final_degrees == degrees,
            "objective": 0.0,
        }
        logger.info(
            "GRAPH_RANDOM_REWIRE_DONE optional_edges=%s attempts=%s accepted=%s degree_sequence_preserved=%s elapsed_seconds=%.1f",
            len(edge_set),
            attempts,
            accepted,
            diagnostics["degree_sequence_preserved"],
            time.perf_counter() - rewiring_started,
        )
        return diagnostics

    @staticmethod
    def _compile_operator_links(
        threads: List[InteractionThread],
        max_neighbors: int = 3,
        min_similarity: float = 0.25,
        candidate_multiplier: int = 3,
        sketch_dim: int = 6,
        objective_scale: float = 1.0,
        swap_search_width: int = 256,
        max_swaps: int = 12,
    ) -> Dict[str, Any]:
        """Compile the final wiring with a finite-horizon operator sketch.

        The compiler is deliberately query-free. Chronology and locator
        embeddings define a candidate edge pool, while a deterministic probe
        sketch estimates the diagonal of a finite-horizon diffusion Gramian.
        Bounded two-switch search maximizes this coverage surrogate under the
        exact undirected edge budget of the initial graph. The sketch is only
        a construction-time objective: no query or answer evidence enters
        this function.
        """
        for thread in threads:
            thread.links = []
        count = len(threads)
        logger = logging.getLogger("paired_graph.graph")
        compiler_started = time.perf_counter()
        logger.info("GRAPH_COMPILER_START cells=%s", count)
        if count <= 1:
            return {
                "mode": "operator-aware",
                "candidate_edges": 0,
                "selected_edges": 0,
                "selected_optional_edges": 0,
                "mandatory_edges": 0,
                "baseline_edge_budget": 0,
                "objective": 0.0,
                "operator_horizon": 3,
                "sketch_dim": max(1, int(sketch_dim)),
                "degree_cap_mode": "baseline-per-node",
                "max_optional_degree_cap": 0,
                "max_selected_optional_degree": 0,
                "degree_cap_satisfied": True,
                "rewiring_swaps": 0,
                "candidate_switches": 0,
                "evaluated_switches": 0,
            }

        max_neighbors = max(1, int(max_neighbors))
        min_similarity = max(-1.0, min(1.0, float(min_similarity)))
        candidate_multiplier = max(1, int(candidate_multiplier))
        sketch_dim = max(1, min(16, int(sketch_dim)))
        objective_scale = max(0.0, float(objective_scale))
        swap_search_width = max(1, int(swap_search_width))
        max_swaps = max(0, int(max_swaps))
        index = {thread.thread_id: position for position, thread in enumerate(threads)}

        def key(left: str, right: str) -> Tuple[str, str]:
            return (left, right) if left < right else (right, left)

        chronology: set[Tuple[str, str]] = set()
        ordered = sorted(threads, key=lambda item: (item.ordinal, item.thread_id))
        for left, right in zip(ordered, ordered[1:]):
            chronology.add(key(left.thread_id, right.thread_id))

        pair_similarity: Dict[Tuple[str, str], float] = {}
        all_pairs: List[Tuple[float, str, str]] = []
        pairwise_started = time.perf_counter()
        report_every = max(1, count // 10)
        for left_position, left in enumerate(threads):
            if left_position % report_every == 0:
                logger.info(
                    "GRAPH_PAIRWISE_PROGRESS rows=%s/%s elapsed_seconds=%.1f",
                    left_position,
                    count,
                    time.perf_counter() - pairwise_started,
                )
            for right in threads[left_position + 1:]:
                similarity = ThreadStore._cosine(left.embedding, right.embedding)
                pair = key(left.thread_id, right.thread_id)
                pair_similarity[pair] = similarity
                all_pairs.append((similarity, left.thread_id, right.thread_id))
        logger.info(
            "GRAPH_PAIRWISE_DONE cells=%s pairs=%s elapsed_seconds=%.1f",
            count,
            len(pair_similarity),
            time.perf_counter() - pairwise_started,
        )

        similarities = list(pair_similarity.values())
        mu_sem = sum(similarities) / max(1, len(similarities))
        sigma_sem = math.sqrt(
            sum((value - mu_sem) ** 2 for value in similarities)
            / max(1, len(similarities))
        )
        semantic_threshold = mu_sem + sigma_sem
        candidate_threshold = mu_sem

        # Reproduce the paper's initial graph edge budget exactly: every
        # semantic pair above mu+sigma plus every globally adjacent sequential
        # edge. This makes rewiring a topology change rather than a density
        # increase.
        baseline_pairs = set(chronology)
        for similarity, left_id, right_id in all_pairs:
            if similarity >= semantic_threshold:
                baseline_pairs.add(key(left_id, right_id))
        baseline_budget = len(baseline_pairs)

        # The candidate pool is wider than the initial semantic graph so the
        # operator objective can choose a relay edge that is not locally the
        # most similar neighbor. All initial edges are always retained as
        # candidates, so the budgeted comparison is well-defined.
        candidate_pairs = set(chronology)
        for similarity, left_id, right_id in all_pairs:
            if similarity >= candidate_threshold:
                candidate_pairs.add(key(left_id, right_id))
        candidate_pairs.update(baseline_pairs)
        if len(candidate_pairs) < baseline_budget:
            for _similarity, left_id, right_id in sorted(
                all_pairs,
                key=lambda item: (-item[0], index[item[1]], index[item[2]]),
            ):
                candidate_pairs.add(key(left_id, right_id))
                if len(candidate_pairs) >= baseline_budget:
                    break

        def deterministic_probe(thread_id: str, column: int) -> float:
            import hashlib
            digest = hashlib.sha256(
                f"memwire-probe-v1|{thread_id}|{column}".encode("utf-8")
            ).digest()
            return 1.0 if digest[0] & 1 else -1.0

        probes = [
            [deterministic_probe(thread.thread_id, column) for column in range(sketch_dim)]
            for thread in threads
        ]
        if sketch_dim >= 1:
            for row in probes:
                row[0] = 1.0
        if sketch_dim >= 2:
            denominator = max(1, count - 1)
            for position, row in enumerate(probes):
                row[1] = 2.0 * position / denominator - 1.0

        horizon = max(0, int(getattr(config, "THREAD_OPERATOR_HORIZON", 3)))
        lambda_bal = float(getattr(config, "THREAD_OPERATOR_LAMBDA_BAL", 1.0 / 40.0))
        epsilon = float(getattr(config, "THREAD_OPERATOR_EPS", 1e-6))

        # Derive an explicit optional-degree budget for every vertex from the
        # initial graph. Chronology is mandatory and is not charged against
        # this budget. The operator-aware compiler may rewire edges, but it
        # cannot make any cell more connected than its matched initial
        # optional degree.
        baseline_optional = baseline_pairs - chronology
        optional_degree_caps: Dict[str, int] = {
            thread.thread_id: 0 for thread in threads
        }
        for left_id, right_id in baseline_optional:
            optional_degree_caps[left_id] += 1
            optional_degree_caps[right_id] += 1

        def optional_degrees(selected_pairs: set[Tuple[str, str]]) -> Dict[str, int]:
            degrees = {thread.thread_id: 0 for thread in threads}
            for left_id, right_id in selected_pairs - chronology:
                degrees[left_id] += 1
                degrees[right_id] += 1
            return degrees

        def weighted_degrees(selected_pairs: set[Tuple[str, str]]) -> List[float]:
            values = [0.0 for _ in threads]
            for left_id, right_id in selected_pairs:
                similarity = pair_similarity.get((left_id, right_id), 0.0)
                weight = 1.0 if (left_id, right_id) in chronology else (similarity + 1.0) / 2.0
                values[index[left_id]] += weight
                values[index[right_id]] += weight
            return values

        target_weighted_degree = sum(weighted_degrees(baseline_pairs)) / max(1, count)

        def objective(selected_pairs: set[Tuple[str, str]]) -> float:
            """Diagonal log-volume surrogate for finite-horizon reachability."""
            adjacency: List[List[Tuple[int, float]]] = [[] for _ in threads]
            degree = [0.0 for _ in threads]
            for left_id, right_id in selected_pairs:
                left_position = index[left_id]
                right_position = index[right_id]
                similarity = pair_similarity.get((left_id, right_id), 0.0)
                if (left_id, right_id) in chronology:
                    weight = 1.0
                else:
                    weight = (similarity + 1.0) / 2.0
                adjacency[left_position].append((right_position, weight))
                adjacency[right_position].append((left_position, weight))
                degree[left_position] += weight
                degree[right_position] += weight

            state = [list(row) for row in probes]
            energy = [
                sum(float(value) * float(value) for value in row)
                for row in state
            ]
            for step in range(horizon):
                next_state = [[0.0 for _ in range(sketch_dim)] for _ in threads]
                for position, neighbors in enumerate(adjacency):
                    if not neighbors:
                        continue
                    for neighbor, weight in neighbors:
                        denominator = math.sqrt(
                            max(1e-9, degree[position])
                            * max(1e-9, degree[neighbor])
                        )
                        coefficient = weight / denominator
                        for column in range(sketch_dim):
                            next_state[position][column] += coefficient * state[neighbor][column]
                state = next_state
                for position, row in enumerate(state):
                    energy[position] += sum(
                        float(value) * float(value) for value in row
                    )
            value = sum(math.log(max(0.0, item) + epsilon) for item in energy)
            balance = sum(
                (item - target_weighted_degree) ** 2
                for item in degree
            ) / max(1, count)
            normalized_balance = balance / (
                count * target_weighted_degree * target_weighted_degree + epsilon
            )
            return objective_scale * (value - lambda_bal * normalized_balance)

        # Start from the initial optional semantic graph so the feasible set is never
        # empty. A one-edge replacement cannot preserve every individual
        # degree: after removing (a, b), only a and b have a free stub. We
        # therefore use degree-preserving two-switches:
        # (a, b), (c, d) -> (a, c), (b, d).
        #
        # Thus every accepted move preserves the complete initial optional
        # degree sequence as well as the global edge count. The finite-horizon
        # operator is the decision objective; the candidate screen below only
        # bounds write-time computation.
        selected = set(chronology) | set(baseline_optional)
        current_value = objective(selected)
        optional_degrees_now = optional_degrees(selected)

        def pair_probe_distance(pair: Tuple[str, str]) -> float:
            left_row = probes[index[pair[0]]]
            right_row = probes[index[pair[1]]]
            return sum(
                (float(left_value) - float(right_value)) ** 2
                for left_value, right_value in zip(left_row, right_row)
            ) / max(1, sketch_dim)

        def feasible_two_switches(
            current_optional: set[Tuple[str, str]],
        ) -> List[Tuple[float, Tuple[str, str], Tuple[str, str], Tuple[str, str], Tuple[str, str]]]:
            # A proposal is (screen score, remove-left, remove-right,
            # add-left, add-right). The screen score is not the objective.
            optional_edges = sorted(
                current_optional,
                key=lambda pair: (index[pair[0]], index[pair[1]]),
            )
            proposals: List[
                Tuple[float, Tuple[str, str], Tuple[str, str], Tuple[str, str], Tuple[str, str]]
            ] = []
            seen: set[
                Tuple[Tuple[str, str], Tuple[str, str], Tuple[str, str], Tuple[str, str]]
            ] = set()
            proposal_started = time.perf_counter()
            report_proposals_every = max(1, len(optional_edges) // 10)
            for left_position, remove_left in enumerate(optional_edges):
                if left_position % report_proposals_every == 0:
                    logger.info(
                        "GRAPH_SWITCH_ENUM_PROGRESS rows=%s/%s optional_edges=%s elapsed_seconds=%.1f",
                        left_position,
                        len(optional_edges),
                        len(optional_edges),
                        time.perf_counter() - proposal_started,
                    )
                a, b = remove_left
                for remove_right in optional_edges[left_position + 1 :]:
                    c, d = remove_right
                    if len({a, b, c, d}) != 4:
                        continue
                    for add_left, add_right in (
                        (key(a, c), key(b, d)),
                        (key(a, d), key(b, c)),
                    ):
                        if (
                            add_left == add_right
                            or add_left not in candidate_pairs
                            or add_right not in candidate_pairs
                            or add_left in chronology
                            or add_right in chronology
                            or add_left in current_optional
                            or add_right in current_optional
                        ):
                            continue
                        normalized_removes = tuple(sorted((remove_left, remove_right)))
                        normalized_adds = tuple(sorted((add_left, add_right)))
                        signature = (
                            normalized_removes[0],
                            normalized_removes[1],
                            normalized_adds[0],
                            normalized_adds[1],
                        )
                        if signature in seen:
                            continue
                        seen.add(signature)
                        screen_score = (
                            pair_probe_distance(add_left)
                            + pair_probe_distance(add_right)
                            + 0.10 * (
                                pair_similarity.get(add_left, -1.0)
                                + pair_similarity.get(add_right, -1.0)
                            )
                        )
                        proposals.append((
                            screen_score,
                            remove_left,
                            remove_right,
                            add_left,
                            add_right,
                        ))
            proposals.sort(
                key=lambda item: (
                    -item[0],
                    index[item[1][0]],
                    index[item[1][1]],
                    index[item[2][0]],
                    index[item[2][1]],
                    index[item[3][0]],
                    index[item[3][1]],
                    index[item[4][0]],
                    index[item[4][1]],
                )
            )
            logger.info(
                "GRAPH_SWITCH_ENUM_DONE proposals=%s elapsed_seconds=%.1f",
                len(proposals),
                time.perf_counter() - proposal_started,
            )
            return proposals

        gains: List[float] = []
        rewiring_swaps = 0
        candidate_switches = 0
        evaluated_switches = 0
        for iteration in range(max_swaps):
            iteration_started = time.perf_counter()
            proposals = feasible_two_switches(selected - chronology)
            candidate_switches += len(proposals)
            logger.info(
                "GRAPH_SEARCH_ITERATION_START iteration=%s/%s proposals=%s evaluated_limit=%s",
                iteration + 1,
                max_swaps,
                len(proposals),
                min(len(proposals), swap_search_width),
            )
            if not proposals:
                break
            best_move: Optional[
                Tuple[Tuple[str, str], Tuple[str, str], Tuple[str, str], Tuple[str, str]]
            ] = None
            best_value = current_value
            best_gain = -math.inf
            last_progress_log = time.perf_counter()
            for proposal_index, (_screen, remove_left, remove_right, add_left, add_right) in enumerate(
                proposals[:swap_search_width],
                1,
            ):
                trial = (selected - {remove_left, remove_right}) | {add_left, add_right}
                trial_degrees = optional_degrees(trial)
                if trial_degrees != optional_degrees_now:
                    continue
                evaluated_switches += 1
                value = objective(trial)
                if (
                    evaluated_switches % 64 == 0
                    or time.perf_counter() - last_progress_log >= 30.0
                ):
                    logger.info(
                        "GRAPH_SEARCH_PROGRESS iteration=%s proposal=%s/%s evaluated=%s elapsed_seconds=%.1f",
                        iteration + 1,
                        proposal_index,
                        min(len(proposals), swap_search_width),
                        evaluated_switches,
                        time.perf_counter() - iteration_started,
                    )
                    last_progress_log = time.perf_counter()
                gain = value - current_value
                # The exact operator score is primary; similarity only
                # resolves numerical ties between degree-preserving moves.
                rank_value = value + 1e-4 * (
                    pair_similarity.get(add_left, -1.0)
                    + pair_similarity.get(add_right, -1.0)
                )
                if rank_value > best_value + 1e-10:
                    best_move = (remove_left, remove_right, add_left, add_right)
                    best_value = rank_value
                    best_gain = gain
            logger.info(
                "GRAPH_SEARCH_ITERATION_DONE iteration=%s evaluated=%s swaps=%s best_gain=%s elapsed_seconds=%.1f",
                iteration + 1,
                evaluated_switches,
                rewiring_swaps,
                best_gain if best_move is not None else 0.0,
                time.perf_counter() - iteration_started,
            )
            if best_move is None or best_gain <= 1e-10:
                break
            remove_left, remove_right, add_left, add_right = best_move
            selected.remove(remove_left)
            selected.remove(remove_right)
            selected.add(add_left)
            selected.add(add_right)
            optional_degrees_now = optional_degrees(selected)
            current_value = objective(selected)
            gains.append(best_gain)
            rewiring_swaps += 1

        def add_edge(left: InteractionThread, right: InteractionThread, relation: str, weight: float) -> None:
            left.links.append(ThreadLink(
                target_thread_id=right.thread_id,
                relation=relation,
                weight=float(weight),
            ))

        for left_id, right_id in sorted(
            selected,
            key=lambda pair: (index[pair[0]], index[pair[1]]),
        ):
            left = threads[index[left_id]]
            right = threads[index[right_id]]
            similarity = pair_similarity.get((left_id, right_id), 0.0)
            if (left_id, right_id) in chronology:
                relation = "chronology-next"
                weight = 1.0
            else:
                relation = "semantic"
                weight = (similarity + 1.0) / 2.0
            add_edge(left, right, relation, weight)
            add_edge(right, left, relation, weight)

        diagnostics = {
            "mode": "operator-aware",
            "candidate_edges": len(candidate_pairs),
            "selected_edges": len(selected),
            "selected_optional_edges": len(selected - chronology),
            "mandatory_edges": len(chronology),
            "baseline_edge_budget": baseline_budget,
            "objective": round(float(current_value), 6),
            "operator_horizon": horizon,
            "sketch_dim": sketch_dim,
            "mean_optional_gain": round(
                sum(gains) / max(1, len(gains)),
                6,
            ),
            "degree_cap_mode": "baseline-per-node",
            "max_optional_degree_cap": max(optional_degree_caps.values(), default=0),
            "max_selected_optional_degree": max(optional_degrees_now.values(), default=0),
            "degree_cap_satisfied": all(
                optional_degrees_now[thread_id] == optional_degree_caps[thread_id]
                for thread_id in optional_degrees_now
            ),
            "rewiring_swaps": rewiring_swaps,
            "candidate_switches": candidate_switches,
            "evaluated_switches": evaluated_switches,
            "cross_session_edges": sum(
                1
                for left_id, right_id in selected
                if threads[index[left_id]].session_id != threads[index[right_id]].session_id
            ),
        }
        logger.info(
            "GRAPH_COMPILER_DONE cells=%s pairs=%s candidates=%s selected_edges=%s swaps=%s elapsed_seconds=%.1f",
            count,
            len(pair_similarity),
            len(candidate_pairs),
            len(selected),
            rewiring_swaps,
            time.perf_counter() - compiler_started,
        )
        return diagnostics

    @staticmethod
    def _thread_to_head_row(thread: InteractionThread) -> Dict[str, Any]:
        return {
            "thread_id": thread.thread_id,
            "head": thread.head,
            "session_id": thread.session_id or "",
            "ordinal": int(thread.ordinal),
            "start_timestamp": thread.start_timestamp or "",
            "end_timestamp": thread.end_timestamp or "",
            "links_json": json.dumps([link.dict() for link in thread.links], ensure_ascii=True),
            "vector": list(thread.embedding),
        }

    @staticmethod
    def _thread_to_payload_row(thread: InteractionThread) -> Dict[str, Any]:
        return {
            "thread_id": thread.thread_id,
            "nodes_json": json.dumps(
                [node.dict() for node in thread.nodes],
                ensure_ascii=True,
            ),
        }

    @staticmethod
    def _thread_to_row(thread: InteractionThread) -> Dict[str, Any]:
        """Compatibility row for the dependency-free JSON backend."""
        row = ThreadStore._thread_to_head_row(thread)
        row["nodes_json"] = json.dumps(
            [node.dict() for node in thread.nodes],
            ensure_ascii=True,
        )
        return row

    @staticmethod
    def _row_to_thread(row: Dict[str, Any]) -> InteractionThread:
        def parse_json(value: Any, default: Any) -> Any:
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return default
            return value if value is not None else default

        return InteractionThread(
            thread_id=str(row["thread_id"]),
            head=str(row.get("head") or ""),
            session_id=str(row.get("session_id") or "") or None,
            ordinal=int(row.get("ordinal") or 0),
            start_timestamp=str(row.get("start_timestamp") or "") or None,
            end_timestamp=str(row.get("end_timestamp") or "") or None,
            nodes=[ThreadNode(**item) for item in parse_json(row.get("nodes_json"), [])],
            links=[ThreadLink(**item) for item in parse_json(row.get("links_json"), [])],
            embedding=[float(value) for value in (row.get("vector") or [])],
        )

    @staticmethod
    def _row_to_head_thread(row: Dict[str, Any]) -> InteractionThread:
        """Decode a LanceDB head row without loading node payloads."""
        def parse_json(value: Any, default: Any) -> Any:
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return default
            return value if value is not None else default

        return InteractionThread(
            thread_id=str(row["thread_id"]),
            head=str(row.get("head") or ""),
            session_id=str(row.get("session_id") or "") or None,
            ordinal=int(row.get("ordinal") or 0),
            start_timestamp=str(row.get("start_timestamp") or "") or None,
            end_timestamp=str(row.get("end_timestamp") or "") or None,
            nodes=[],
            links=[ThreadLink(**item) for item in parse_json(row.get("links_json"), [])],
            embedding=[float(value) for value in (row.get("vector") or [])],
        )

    def _ensure_payload_index(self) -> Dict[str, Dict[str, Any]]:
        """Materialize the small payload table once for local expansion.

        Payload rows are compact source JSON and contain no ANN vectors. A
        single sequential read per root is substantially cheaper than issuing
        one LanceDB point query for every traversed edge and is equivalent to
        the old lookup semantics.
        """
        if self._payload_index is not None:
            return self._payload_index
        with self._metadata_index_lock:
            if self._payload_index is not None:
                return self._payload_index
            if self._use_lancedb:
                try:
                    rows = self.payload_table.to_arrow().to_pylist()
                except Exception:
                    rows = []
                    self._payload_index_load_failed = True
            else:
                rows = list(self._rows)
            self._payload_index = {
                str(row.get("thread_id")): row
                for row in rows
                if row.get("thread_id") is not None
            }
            return self._payload_index

    def _ensure_head_index(self) -> Dict[str, Dict[str, Any]]:
        """Materialize lightweight head metadata once for edge traversal."""
        if self._head_index is not None:
            return self._head_index
        with self._metadata_index_lock:
            if self._head_index is not None:
                return self._head_index
            if self._use_lancedb:
                try:
                    rows = self.table.to_arrow().to_pylist()
                except Exception:
                    rows = []
                    self._head_index_load_failed = True
            else:
                rows = list(self._rows)
            self._head_index = {
                str(row.get("thread_id")): row
                for row in rows
                if row.get("thread_id") is not None
            }
            return self._head_index

    def _load_payload_row(self, thread_id: str) -> Optional[Dict[str, Any]]:
        row = self._ensure_payload_index().get(str(thread_id))
        if row is None and self._use_lancedb and self._payload_index_load_failed:
            # Preserve compatibility with LanceDB versions that cannot expose
            # an Arrow snapshot; this path is only a failure fallback.
            try:
                escaped = str(thread_id).replace("'", "''")
                rows = (
                    self.payload_table.search()
                    .where(f"thread_id = '{escaped}'", prefilter=True)
                    .limit(1)
                    .to_list()
                )
                row = rows[0] if rows else None
            except Exception:
                row = None
        if row is not None:
            self._payload_cache[str(thread_id)] = row
        return row

    def _load_head_thread(self, thread_id: str) -> Optional[InteractionThread]:
        """Load one lightweight head row for a graph edge target."""
        if thread_id in self._head_cache:
            return self._head_cache[thread_id]
        row = self._ensure_head_index().get(str(thread_id))
        if row is None and self._use_lancedb and self._head_index_load_failed:
            try:
                escaped = str(thread_id).replace("'", "''")
                rows = (
                    self.table.search()
                    .where(f"thread_id = '{escaped}'", prefilter=True)
                    .limit(1)
                    .to_list()
                )
                row = rows[0] if rows else None
            except Exception:
                row = None
        if row is None:
            return None
        header = self._row_to_head_thread(row)
        self._head_cache[thread_id] = header
        return header

    def _hydrate_thread(
        self,
        head_row: Dict[str, Any],
        payload_row: Optional[Dict[str, Any]] = None,
    ) -> InteractionThread:
        if not self._use_lancedb:
            thread_id = str(head_row.get("thread_id"))
            cached = self._thread_cache.get(thread_id)
            return cached.copy(deep=True) if cached is not None else self._row_to_thread(head_row)
        header = self._row_to_head_thread(head_row)
        payload_row = payload_row or self._load_payload_row(header.thread_id)
        if payload_row:
            raw_nodes = payload_row.get("nodes_json")
            if isinstance(raw_nodes, str):
                try:
                    raw_nodes = json.loads(raw_nodes)
                except json.JSONDecodeError:
                    raw_nodes = []
            header.nodes = [ThreadNode(**item) for item in (raw_nodes or [])]
        return header

    def _hydrate_thread_from_header(self, header: InteractionThread) -> InteractionThread:
        if not self._use_lancedb:
            cached = self._thread_cache.get(header.thread_id)
            return cached.copy(deep=True) if cached is not None else header
        if header.nodes:
            return header
        row = self._load_payload_row(header.thread_id)
        if row:
            raw_nodes = row.get("nodes_json")
            if isinstance(raw_nodes, str):
                try:
                    raw_nodes = json.loads(raw_nodes)
                except json.JSONDecodeError:
                    raw_nodes = []
            header.nodes = [ThreadNode(**item) for item in (raw_nodes or [])]
        return header
