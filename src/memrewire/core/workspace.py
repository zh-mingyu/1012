"""Transient query-time assembly over a fixed memory substrate.

The persistent store is source-faithful and read-only during a query. A single
address lookup supplies a seed, then one shared local transition rule is
rolled out over the fixed substrate. The answer context is read from the
resulting stable activation field through a budgeted marginal readout. No
query-specific roles or topology are written back to memory.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, PrivateAttr

from memwire.core.models.memory_entry import InteractionThread
from memwire.core.thread_store import ThreadCandidate


class WorkspaceEvidence(BaseModel):
    """A source cell exposed by one transient activation field.

    All state diagnostics are query-local.  The continuous role coordinates
    are generated during the recurrent update and are exposed here for the
    answer model; the short ``role`` label is only a compatibility projection.
    """

    evidence_id: str
    thread_id: str
    node_id: str
    text: str
    thread_head: str = ""
    source_turn_ids: List[str] = Field(default_factory=list)
    role: str = "context"
    contribution: str = "context"
    role_gain: float = 0.0
    role_components: Dict[str, float] = Field(default_factory=dict)
    route_score: float = 0.0
    node_score: float = 0.0
    activation_score: float = 0.0
    seed_affinity: float = 0.0
    support_influence: float = 0.0
    inhibition_score: float = 0.0
    state_stability: float = 0.0
    readout_score: float = 0.0
    # Canonical functional representation. These coordinates are inferred
    # from the query-local field and are never persisted in ThreadStore.
    functional_role: str = "context"
    role_vector: Dict[str, float] = Field(default_factory=dict)
    query_drive: float = 0.0
    local_support: float = 0.0
    relay_mediation: float = 0.0
    marginal_gain: float = 0.0
    thread_ordinal: int = 0
    thread_rank: int = 0
    ordinal: int = 0
    timestamp: Optional[str] = None


class WorkspaceRelation(BaseModel):
    source_evidence_id: str
    target_evidence_id: str
    relation_type: str
    anchors: List[str] = Field(default_factory=list)


class MemoryWorkspace(BaseModel):
    """Ephemeral query-conditioned state handed to the answer model."""

    workspace_id: str
    query: str
    address: str = ""
    question_type: str = "general"
    complexity: str = "LOW"
    evidence: List[WorkspaceEvidence] = Field(default_factory=list)
    relations: List[WorkspaceRelation] = Field(default_factory=list)
    sufficient: bool = False
    assembly_mode: str = "transient-assembly"
    inference_family: str = "transient-role-binding"
    source_candidate_count: int = 0
    routed_thread_count: int = 0
    expanded_thread_count: int = 0
    node_candidate_count: int = 0
    visible_span_count: int = 0
    routed_source_turn_ids: List[str] = Field(default_factory=list)
    expanded_source_turn_ids: List[str] = Field(default_factory=list)
    activation_total: float = 0.0
    activation_entropy: float = 0.0
    activation_iterations: int = 0
    graph_vertex_count: int = 0
    graph_edge_count: int = 0
    # Construction-time wiring diagnostics.  These are copied from the
    # persistent store and are independent of the query-local field.
    topology_mode: str = "operator-aware"
    topology_candidate_edges: int = 0
    topology_selected_edges: int = 0
    topology_optional_edges: int = 0
    topology_mandatory_edges: int = 0
    topology_edge_budget: int = 0
    topology_operator_objective: float = 0.0
    topology_mean_gain: float = 0.0
    topology_cross_session_edges: int = 0
    global_route_calls: int = 1
    address_planner_calls: int = 0
    indexed_thread_count: int = 0
    local_edge_visits: int = 0
    # Address and receptive-field diagnostics.
    route_probe_count: int = 0
    route_width: int = 0
    receptive_field_scope: int = 0
    adaptive_route: bool = False
    routed_thread_ids: List[str] = Field(default_factory=list)
    expanded_thread_ids: List[str] = Field(default_factory=list)
    candidate_thread_ids: List[str] = Field(default_factory=list)
    candidate_node_ids: List[str] = Field(default_factory=list)
    visible_node_ids: List[str] = Field(default_factory=list)
    role_binding_mode: str = "field-coupled"
    role_counts: Dict[str, int] = Field(default_factory=dict)
    # Primary dynamics diagnostics.
    dynamics_mode: str = "local-recurrent-assembly"
    rollout_steps: int = 0
    rollout_converged: bool = False
    state_delta: float = 0.0
    activation_energy: float = 0.0
    # The following fields describe the transient field and the external
    # serialization safeguard.  They are not part of state inference.
    prompt_guard_tokens: int = 0
    visibility_cutoff: float = 0.0
    field_level: float = 0.0
    readout_rule: str = "relative-field-level"
    readout_capacity_cells: int = 0
    readout_budget_tokens: int = 0
    readout_utility: float = 0.0
    serialized_cell_count: int = 0
    assembly_coverage: float = 0.0
    support_density: float = 0.0
    message_edge_visits: int = 0
    seed_count: int = 0
    _legacy_item_guard: Optional[int] = PrivateAttr(default=None)

    @property
    def entries(self) -> List[WorkspaceEvidence]:
        return list(self.evidence)

    @property
    def assembly_cells(self) -> List[WorkspaceEvidence]:
        """Canonical name for the cells exposed by the transient field."""
        return list(self.evidence)

    @property
    def assembly_id(self) -> str:
        """Canonical identifier for one query-local state."""
        return self.workspace_id

    @property
    def role_bound_cells(self) -> List[WorkspaceEvidence]:
        """Canonical view of cells carrying transient role coordinates."""
        return list(self.evidence)

    @property
    def selected_thread_count(self) -> int:
        """Legacy alias for the number of visible source spans."""
        return self.visible_span_count

    @property
    def selected_node_ids(self) -> List[str]:
        """Legacy alias for visible source-cell identifiers."""
        return list(self.visible_node_ids)

    @property
    def route_budget(self) -> int:
        """Compatibility alias for the address probe width."""
        return self.route_width

    @property
    def expansion_budget(self) -> int:
        """Compatibility alias for the receptive-field scope."""
        return self.receptive_field_scope

    def __len__(self) -> int:
        return len(self.evidence)

    def __bool__(self) -> bool:
        return bool(self.evidence)

    # Compatibility views for older evaluators.  They are properties rather
    # than stored state, so the serialized schema describes the assembly state.
    @property
    def coverage_mode(self) -> str:
        return self.assembly_mode

    @property
    def selection_budget(self) -> int:
        """Compatibility metric for an explicitly requested item guard."""
        return int(self._legacy_item_guard or 0)

    @property
    def readout_limit(self) -> int:
        """Deprecated compatibility view; zero means no explicit item guard."""
        return int(self._legacy_item_guard or 0)

    @property
    def readout_threshold(self) -> float:
        """Deprecated compatibility view for the field level."""
        return self.field_level

    @property
    def selection_coverage(self) -> float:
        return self.assembly_coverage

    @property
    def selection_rounds(self) -> int:
        """Legacy metric alias for the number of visible cells."""
        return len(self.evidence)

    @property
    def selected_activation_mass(self) -> float:
        return self.activation_total

    @property
    def transport_flow_units(self) -> int:
        """Legacy metric alias for local message-edge visits."""
        return self.message_edge_visits

    @property
    def transport_cost(self) -> float:
        """Legacy metric alias for the activation-energy diagnostic."""
        return self.activation_energy


class TransientRoleBindingBuilder:
    """Infer a query-local assembly with one shared recurrent operator.

    The builder keeps the source substrate fixed.  A query supplies an initial
    address signal, local messages evolve that signal on the stored graph, and
    the final activation field is read out by a capacity-aware marginal rule.
    A separate serializer guard protects the answer prompt; it is not part of
    the state dynamics.
    """

    def __init__(
        self,
        item_guard: Optional[int] = None,
        readout_limit: Optional[int] = None,
        min_readout: int = 1,
        max_relations: int = 12,
        node_seed_weight: float = 0.80,
        rollout_steps: int = 20,
        step_size: float = 0.35,
        convergence_tol: float = 1e-3,
        readout_floor: float = 0.30,
        visibility_floor: Optional[float] = None,
        prompt_guard_tokens: int = 0,
        max_readout_cells: Optional[int] = 40,
        readout_token_budget: int = 0,
        readout_mode: str = "field",
        compatibility_scale: float = 1.35,
        inhibition_scale: float = 0.35,
        **legacy_options: Any,
    ) -> None:
        # Accept historical keyword names at the boundary so old benchmark
        # launchers remain runnable. They are translated once and never enter
        # the recurrent update.
        # ``readout_limit`` and ``max_evidence`` are legacy serializer guards;
        # the field readout itself never optimizes or targets an item count.
        if item_guard is None:
            item_guard = readout_limit
        if item_guard is None and "max_evidence" in legacy_options:
            item_guard = legacy_options["max_evidence"]
        if "min_evidence" in legacy_options:
            min_readout = legacy_options["min_evidence"]
        self._legacy_item_guard = (
            max(1, int(item_guard)) if item_guard is not None else None
        )
        self.min_readout = max(1, int(min_readout))
        self.max_relations = max(1, int(max_relations))
        self.diagnostic_epsilon = 1e-9
        self.node_seed_weight = max(0.05, min(0.95, float(node_seed_weight)))
        self.head_seed_weight = max(0.05, 1.0 - self.node_seed_weight)
        self.rollout_steps = max(1, min(64, int(rollout_steps)))
        self.step_size = max(0.05, min(0.95, float(step_size)))
        self.convergence_tol = max(1e-7, float(convergence_tol))
        if visibility_floor is None:
            visibility_floor = readout_floor
        self.visibility_floor = max(0.05, min(0.90, float(visibility_floor)))
        self.prompt_guard_tokens = max(0, int(prompt_guard_tokens))
        if max_readout_cells is None and "max_visible_cells" in legacy_options:
            max_readout_cells = legacy_options["max_visible_cells"]
        self.max_readout_cells = (
            max(1, int(max_readout_cells)) if max_readout_cells is not None else 40
        )
        self.readout_token_budget = max(0, int(readout_token_budget))
        self.readout_mode = str(readout_mode or "field").strip().lower()
        if self.readout_mode not in {"field", "query_similarity"}:
            raise ValueError(f"unsupported readout_mode: {readout_mode}")
        self.compatibility_scale = max(0.05, float(compatibility_scale))
        self.inhibition_scale = max(0.0, float(inhibition_scale))

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
        right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
        if not left_norm or not right_norm:
            return 0.0
        return max(-1.0, min(1.0, dot / (left_norm * right_norm)))

    @staticmethod
    def _temporal_signal(text: str) -> float:
        """Estimate whether a node explicitly exposes temporal language.

        This is a local observability feature, not a retrieval query.  It lets
        the role-conditioned kernel preserve turns that carry relative dates
        even when their semantic embeddings are dissimilar to the answer
        turn (for example, a response saying ``the Friday before ...``).
        """
        value = str(text or "").lower()
        markers = re.findall(
            r"\b(?:yesterday|today|tomorrow|last|next|before|after|"
            r"week|weekend|month|year|monday|tuesday|wednesday|thursday|"
            r"friday|saturday|sunday|january|february|march|april|may|"
            r"june|july|august|september|october|november|december|\d{4})\b",
            value,
        )
        return min(1.0, 0.5 * len(markers))

    @staticmethod
    def _normalise(values: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(0.0, value) for value in values.values())
        if not total:
            size = max(1, len(values))
            return {key: 1.0 / size for key in values}
        return {key: max(0.0, value) / total for key, value in values.items()}

    @staticmethod
    def _infer_question_type(query: str) -> Tuple[str, str]:
        """Infer rendering/organization hints without creating a search query.

        The evaluator's categories are not exposed to the system.  These
        conservative lexical cues only choose how an already-built workspace
        is presented to the answer model; they never alter the global route or
        add a retrieval channel.
        """
        text = str(query or "").lower()
        if re.search(
            r"\b(when|what date|what year|which year|how long|how many years|"
            r"what month|what day|first time|last time|recently)\b",
            text,
        ):
            return "temporal", "MEDIUM"
        if (
            re.search(r"\bhow many times\b", text)
            or re.search(
                r"\bwhat (activities|events|books|places|types|kinds|names|"
                r"pets|instruments|artists|bands|symbols|ways|things|subjects)\b",
                text,
            )
            or re.search(r"\bwhere (has|have|did)\b", text)
        ):
            return "set-valued", "HIGH"
        if (
            re.search(r"\bboth\b|\bin common\b|\beach other\b", text)
            or re.search(r"\bwhat .* and .* (both|have|did)\b", text)
        ):
            return "multi-hop", "HIGH"
        if re.search(
            r"\b(would|could|likely|might|personality|political|future|"
            r"interested|prefer|open to|potentially)\b",
            text,
        ):
            return "open-domain", "HIGH"
        return "general", "LOW"

    @classmethod
    def _plan_state(
        cls,
        plan: Optional[Dict[str, Any]],
        query: str = "",
    ) -> Tuple[str, str]:
        """Read metadata for rendering; never derive a new search request."""
        plan = plan if isinstance(plan, dict) else {}
        question_type = str(plan.get("question_type") or "general").strip().lower()
        complexity = str(plan.get("complexity") or "LOW").upper()
        if question_type in ("", "general"):
            return cls._infer_question_type(query)
        return question_type, complexity

    def build(
        self,
        query: str,
        address: str,
        query_vector: Sequence[float],
        routes: Sequence[Any],
        expanded: Sequence[ThreadCandidate],
        plan: Optional[Dict[str, Any]] = None,
    ) -> MemoryWorkspace:
        question_type, complexity = self._plan_state(plan, query)
        key = "|".join([query, address, *(item.thread.thread_id for item in expanded)])
        workspace_id = "ws-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        candidates = self._flatten(expanded, query_vector)
        routed_sources = sorted(
            {
                source_id
                for route in routes
                for node in route.thread.nodes
                for source_id in node.source_turn_ids
            }
        )
        expanded_sources = sorted(
            {
                source_id
                for item in expanded
                for node in item.thread.nodes
                for source_id in node.source_turn_ids
            }
        )
        candidate_thread_ids = list(dict.fromkeys(
            str(item["thread_id"]) for item in candidates
        ))
        candidate_node_ids = [str(item["node_id"]) for item in candidates]
        base = dict(
            workspace_id=workspace_id,
            query=query,
            address=address,
            question_type=question_type,
            complexity=complexity,
            source_candidate_count=len(expanded),
            routed_thread_count=len(routes),
            expanded_thread_count=len(expanded),
            prompt_guard_tokens=self.prompt_guard_tokens,
            routed_source_turn_ids=routed_sources,
            expanded_source_turn_ids=expanded_sources,
            global_route_calls=1,
            routed_thread_ids=[route.thread.thread_id for route in routes],
            expanded_thread_ids=[item.thread.thread_id for item in expanded],
            candidate_thread_ids=candidate_thread_ids,
            candidate_node_ids=candidate_node_ids,
            readout_rule=(
                "query-similarity-top-k"
                if self.readout_mode == "query_similarity"
                else "budgeted-marginal-coverage"
            ),
            readout_capacity_cells=self.max_readout_cells,
            readout_budget_tokens=self.readout_token_budget,
        )
        if not candidates:
            workspace = MemoryWorkspace(**base, sufficient=False)
            workspace._legacy_item_guard = self._legacy_item_guard
            return workspace

        if self.readout_mode == "query_similarity":
            adjacency: Dict[str, Dict[str, float]] = {}
            iterations = 0
            edge_visits = 0
            activation_energy = 0.0
            rollout_converged = False
            state_delta = 0.0
            for candidate in candidates:
                candidate["activation_score"] = self._clamp(
                    candidate.get("query_affinity", 0.0)
                )
        else:
            adjacency, seed = self._activation_graph(expanded, candidates)
            (
                node_state,
                iterations,
                edge_visits,
                activation_energy,
                rollout_converged,
                state_delta,
            ) = self._recurrent_rollout(adjacency, seed, candidates)
            for candidate in candidates:
                candidate["activation_score"] = node_state[candidate["node_id"]]

        selected, assembly_coverage, readout_threshold = self._readout(
            candidates,
            question_type,
        )
        scores = [float(item.get("_readout_score", 0.0)) for item in selected]
        selected_with_scores = list(zip(selected, scores))
        role_counts: Dict[str, int] = defaultdict(int)
        evidence = [
            WorkspaceEvidence(
                evidence_id=f"W{index}",
                thread_id=item["thread_id"],
                node_id=item["node_id"],
                text=item["text"],
                thread_head=item["thread_head"],
                source_turn_ids=item["source_turn_ids"],
                role=item.get("role", "context"),
                # Preserve the old temporal rendering label as a compatibility
                # hint; the label is derived after the state is read out.
                contribution=(
                    "temporal-anchor"
                    if "temporal" in question_type.lower() and item.get("timestamp")
                    else item.get("role", "context")
                ),
                functional_role=item.get("role", "context"),
                role_vector={
                    key: round(float(value), 6)
                    for key, value in (item.get("_role_vector") or {}).items()
                },
                query_drive=round(float(item.get("_query_drive", 0.0)), 6),
                local_support=round(float(item.get("_local_support", 0.0)), 6),
                relay_mediation=round(float(item.get("_relay_mediation", 0.0)), 6),
                role_gain=round(float(item.get("role_gain", score)), 6),
                role_components={
                    key: round(float(value), 6)
                    for key, value in (item.get("role_components") or {}).items()
                },
                route_score=round(item["route_score"], 6),
                node_score=round(item["dense_score"], 6),
                activation_score=round(item["activation_score"], 6),
                seed_affinity=round(float(item.get("_seed_signal", 0.0)), 6),
                support_influence=round(float(item.get("_support_influence", 0.0)), 6),
                inhibition_score=round(float(item.get("_inhibition_score", 0.0)), 6),
                state_stability=round(float(item.get("_state_stability", 0.0)), 6),
                readout_score=round(float(item.get("_readout_score", score)), 6),
                # Compatibility field: mirrors the causal role state already
                # produced by the rollout.
                marginal_gain=round(float(item.get("role_gain", score)), 6),
                thread_ordinal=int(item.get("thread_ordinal", 0)),
                thread_rank=item["thread_rank"],
                ordinal=item["ordinal"],
                timestamp=item["timestamp"],
            )
            for index, (item, score) in enumerate(selected_with_scores, 1)
        ]
        for item in evidence:
            role_counts[item.role] += 1
        relations = self._relations(evidence, expanded)
        state_values = [max(0.0, float(item["activation_score"])) for item in candidates]
        state_total = sum(state_values) or 1.0
        state_distribution = [value / state_total for value in state_values]
        entropy = -sum(value * math.log(value) for value in state_distribution if value > 0.0)
        activation_total = sum(item["activation_score"] for item in selected)
        sufficient = bool(evidence) and (
            self.readout_mode == "query_similarity"
            or rollout_converged
            or state_delta <= max(self.convergence_tol * 4.0, 1e-3)
        )
        positive_edges = sum(
            1
            for item in candidates
            if float(item.get("_support_influence", 0.0)) > self.diagnostic_epsilon
        )
        workspace = MemoryWorkspace(
            **base,
            evidence=evidence,
            relations=relations,
            sufficient=sufficient,
            assembly_mode=(
                "query-similarity-readout"
                if self.readout_mode == "query_similarity"
                else "transient-assembly"
            ),
            inference_family=(
                "single-vector-readout"
                if self.readout_mode == "query_similarity"
                else "transient-role-binding"
            ),
            node_candidate_count=len(candidates),
        visible_span_count=len({item["thread_id"] for item, _ in selected_with_scores}),
            assembly_coverage=round(assembly_coverage, 6),
            activation_total=round(activation_total, 6),
            activation_entropy=round(entropy, 6),
            activation_iterations=iterations,
            graph_vertex_count=len(adjacency),
            graph_edge_count=sum(len(edges) for edges in adjacency.values()) // 2,
            local_edge_visits=edge_visits,
            visible_node_ids=[item["node_id"] for item, _ in selected_with_scores],
            role_binding_mode=(
                "query-similarity-only"
                if self.readout_mode == "query_similarity"
                else "field-coupled"
            ),
            role_counts=dict(role_counts),
            dynamics_mode=(
                "not-run-query-similarity"
                if self.readout_mode == "query_similarity"
                else "local-recurrent-assembly"
            ),
            rollout_steps=iterations,
            rollout_converged=rollout_converged,
            state_delta=round(state_delta, 8),
            activation_energy=round(activation_energy, 6),
            visibility_cutoff=round(readout_threshold, 6),
            field_level=round(readout_threshold, 6),
            readout_utility=round(
                sum(float(item.get("_marginal_utility", 0.0)) for item, _ in selected_with_scores),
                6,
            ),
            serialized_cell_count=len(evidence),
            support_density=round(
                sum(float(item.get("_support_density", 0.0)) for item in candidates)
                / max(1, len(candidates)),
                6,
            ),
            message_edge_visits=edge_visits,
            seed_count=sum(
                1 for item in candidates if float(item.get("_seed_signal", 0.0)) >= readout_threshold
            ),
        )
        workspace._legacy_item_guard = self._legacy_item_guard
        return workspace

    @staticmethod
    def _timestamp_key(value: Any) -> Tuple[int, str]:
        """Return a stable sortable key for ISO-like conversation dates."""
        text = str(value or "")
        if not text:
            return (1, "")
        try:
            return (0, datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat())
        except (TypeError, ValueError):
            return (0, text)

    @classmethod
    def _presentation_order(
        cls,
        selected: Sequence[Dict[str, Any]],
        scores: Sequence[float],
        question_type: str,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Order selected source turns by their original global position."""
        pairs = list(zip(selected, scores))
        if not pairs:
            return []

        return sorted(
            pairs,
            key=lambda pair: (
                int(pair[0].get("source_ordinal", 0)),
                int(pair[0].get("thread_ordinal", 0)),
                int(pair[0].get("ordinal", 0)),
                str(pair[0].get("node_id", "")),
            ),
        )

    def _flatten(
        self,
        expanded: Sequence[ThreadCandidate],
        query_vector: Sequence[float],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for thread_rank, item in enumerate(expanded, 1):
            thread: InteractionThread = item.thread
            for node in thread.nodes:
                dense = (self._cosine(query_vector, node.embedding) + 1.0) / 2.0
                candidates.append(
                    {
                        "thread_id": thread.thread_id,
                        "node_id": node.node_id,
                        "text": node.text,
                        "thread_head": thread.head,
                        "source_turn_ids": list(node.source_turn_ids),
                        "ordinal": int(node.ordinal),
                        "thread_ordinal": int(thread.ordinal),
                        "source_ordinal": int(node.source_ordinal or node.ordinal),
                        "timestamp": node.timestamp,
                        "dense_score": dense,
                        "route_score": max(0.0, min(1.0, item.route_score)),
                        "hop": int(item.hop),
                        "thread_rank": thread_rank,
                        "activation_score": 0.0,
                        "query_affinity": dense,
                        # Kept on the transient record for local message
                        # computation; it is never written back to storage.
                        "node_embedding": list(node.embedding),
                    }
                )
        return candidates

    @staticmethod
    def _head_vertex(thread_id: str) -> str:
        return f"h:{thread_id}"

    @staticmethod
    def _node_vertex(node_id: str) -> str:
        return f"n:{node_id}"

    def _activation_graph(
        self,
        expanded: Sequence[ThreadCandidate],
        candidates: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        adjacency: Dict[str, Dict[str, float]] = defaultdict(dict)
        seed: Dict[str, float] = {}
        by_thread: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        item_by_thread = {item.thread.thread_id: item for item in expanded}
        included = set(item_by_thread)

        for candidate in candidates:
            by_thread[candidate["thread_id"]].append(candidate)

        def connect(left: str, right: str, weight: float) -> None:
            weight = max(1e-9, float(weight))
            adjacency[left][right] = max(weight, adjacency[left].get(right, 0.0))
            adjacency[right][left] = max(weight, adjacency[right].get(left, 0.0))

        for thread_id, nodes in by_thread.items():
            nodes.sort(key=lambda item: (item["ordinal"], item["node_id"]))
            head_vertex = self._head_vertex(thread_id)
            adjacency.setdefault(head_vertex, {})
            route_score = item_by_thread[thread_id].route_score
            seed[head_vertex] = self.head_seed_weight * max(1e-6, route_score)
            for node in nodes:
                node_vertex = self._node_vertex(node["node_id"])
                adjacency.setdefault(node_vertex, {})
                connect(head_vertex, node_vertex, 0.35)
                # Only the spans returned by the single address lookup receive
                # an initial cell signal. Expanded neighbors start quiescent
                # and can enter the assembly only through local recurrence.
                seed[node_vertex] = (
                    self.node_seed_weight
                    * max(1e-6, route_score)
                    * max(1e-6, node["dense_score"])
                    / max(1, len(nodes))
                    if int(node.get("hop", 0)) == 0
                    else 0.0
                )
            for left, right in zip(nodes, nodes[1:]):
                connect(
                    self._node_vertex(left["node_id"]),
                    self._node_vertex(right["node_id"]),
                    1.0,
                )

        for item in expanded:
            left = self._head_vertex(item.thread.thread_id)
            for link in item.thread.links:
                if link.target_thread_id in included:
                    connect(
                        left,
                        self._head_vertex(link.target_thread_id),
                        max(0.05, link.weight),
                    )
        return dict(adjacency), self._normalise(seed)

    @staticmethod
    def _sigmoid(value: float) -> float:
        """Numerically stable logistic non-linearity for the cell update."""
        value = float(value)
        if value >= 0.0:
            z = math.exp(-min(value, 60.0))
            return 1.0 / (1.0 + z)
        z = math.exp(max(value, -60.0))
        return z / (1.0 + z)

    @staticmethod
    def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(upper, float(value)))

    def _pair_compatibility(
        self,
        left_vertex: str,
        right_vertex: str,
        edge_weight: float,
        candidate_by_vertex: Dict[str, Dict[str, Any]],
        affinity: Dict[str, float],
    ) -> float:
        """Compute a signed query-conditioned message weight on a fixed edge.

        The edge set comes entirely from the persistent substrate. Query
        conditioning changes only the value of the message, so this operation
        cannot create a task-specific topology.
        """
        left = candidate_by_vertex.get(left_vertex)
        right = candidate_by_vertex.get(right_vertex)
        if left is not None and right is not None:
            semantic = max(
                0.0,
                self._cosine(
                    left.get("node_embedding") or [],
                    right.get("node_embedding") or [],
                ),
            )
        elif left is None and right is None:
            semantic = 1.0
        else:
            node = left or right or {}
            semantic = 0.35 + 0.65 * float(affinity.get(
                left_vertex if left is not None else right_vertex,
                node.get("query_affinity", 0.0),
            ))
        pair_gate = math.sqrt(
            max(0.0, affinity.get(left_vertex, 0.0))
            * max(0.0, affinity.get(right_vertex, 0.0))
        )
        relation = self._clamp(edge_weight, 0.0, 1.0)
        supportive = relation * (0.20 + 0.80 * semantic) * (
            0.35 + 0.65 * pair_gate
        )
        inhibitory = relation * self.inhibition_scale * (1.0 - pair_gate)
        return self.compatibility_scale * supportive - inhibitory

    def _recurrent_rollout(
        self,
        adjacency: Dict[str, Dict[str, float]],
        seed: Dict[str, float],
        candidates: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, float], int, int, float, bool, float]:
        """Roll one shared local transition rule over the fixed substrate.

        The returned node values are transient activations.  Heads and cells
        participate in the same rollout; only cell vertices are exposed to the
        answer model.  The active path consists only of local state updates.
        """
        candidate_by_vertex = {
            self._node_vertex(item["node_id"]): item for item in candidates
        }
        affinity: Dict[str, float] = {}
        vertex_seed: Dict[str, float] = {}
        for vertex in adjacency:
            if vertex in candidate_by_vertex:
                item = candidate_by_vertex[vertex]
                dense = self._clamp(item.get("query_affinity", 0.0))
                route = self._clamp(item.get("route_score", 0.0))
                affinity[vertex] = dense
                # The activation graph, rather than raw similarity, defines
                # the initial state. Non-routed cells therefore begin at zero.
                value = self._clamp(seed.get(vertex, 0.0))
            else:
                value = self._clamp(seed.get(vertex, 0.0))
                affinity[vertex] = value
            vertex_seed[vertex] = self._clamp(value)

        state = dict(vertex_seed)
        # Role channels are part of the transient state transition.  They are
        # recomputed from the current local messages at every step, so the
        # functional interpretation can influence activation instead of being
        # attached only after field exposure.
        role_state: Dict[str, Dict[str, float]] = {}

        def role_channels(
            vertex: str,
            weighted_messages: Sequence[Tuple[str, float]],
            snapshot: Dict[str, float],
        ) -> Dict[str, float]:
            """Return direct, intra-span, and cross-span transient channels."""
            current = candidate_by_vertex.get(vertex)
            current_thread = (
                str(current.get("thread_id", ""))
                if current is not None
                else vertex[2:]
                if vertex.startswith("h:")
                else ""
            )
            direct = self._clamp(
                0.70 * affinity.get(vertex, 0.0)
                + 0.30 * vertex_seed.get(vertex, 0.0)
            )
            local = 0.0
            relay = 0.0
            normalizer = sum(abs(value) for _neighbor, value in weighted_messages)
            for neighbor, compatibility in weighted_messages:
                if compatibility <= 0.0:
                    continue
                contribution = compatibility * self._clamp(snapshot.get(neighbor, 0.0))
                neighbor_item = candidate_by_vertex.get(neighbor)
                neighbor_thread = (
                    str(neighbor_item.get("thread_id", ""))
                    if neighbor_item is not None
                    else neighbor[2:]
                    if neighbor.startswith("h:")
                    else ""
                )
                if current_thread and neighbor_thread == current_thread:
                    local += contribution
                else:
                    relay += contribution
            denominator = normalizer if normalizer > 1e-9 else 1.0
            return {
                "query_drive": direct,
                "local_support": self._clamp(local / denominator),
                "relay_mediation": self._clamp(relay / denominator),
            }

        converged = False
        final_delta = math.inf
        edge_visits = 0
        steps = 0
        for step in range(self.rollout_steps):
            updated: Dict[str, float] = {}
            for vertex in sorted(adjacency):
                neighbors = adjacency.get(vertex) or {}
                weighted_messages: List[Tuple[str, float]] = []
                normalizer = 0.0
                for neighbor, weight in sorted(neighbors.items()):
                    compatibility = self._pair_compatibility(
                        vertex,
                        neighbor,
                        weight,
                        candidate_by_vertex,
                        affinity,
                    )
                    weighted_messages.append((neighbor, compatibility))
                    normalizer += abs(compatibility)
                edge_visits += len(weighted_messages)
                if normalizer > 1e-9:
                    message = sum(
                        compatibility * state.get(neighbor, 0.0)
                        for neighbor, compatibility in weighted_messages
                    ) / normalizer
                    neighborhood_mean = sum(
                        state.get(neighbor, 0.0) for neighbor, _ in weighted_messages
                    ) / max(1, len(weighted_messages))
                else:
                    message = 0.0
                    neighborhood_mean = 0.0
                channels = role_channels(vertex, weighted_messages, state)
                role_state[vertex] = channels
                seed_signal = vertex_seed.get(vertex, 0.0)
                # The negative offset keeps unseeded cells quiet; recurrent
                # support can still lift a neighbor into the assembly.
                bias = -0.85 + 2.55 * seed_signal + 0.55 * seed_signal * seed_signal
                # A role is a small, causal modulation of the field.  The
                # coefficients are fixed across cells; they only distinguish
                # direct drive, local support, and cross-span relay evidence.
                role_bias = (
                    0.08 * channels["query_drive"]
                    + 0.12 * channels["local_support"]
                    + 0.18 * channels["relay_mediation"]
                )
                preactivation = (
                    bias
                    + 1.65 * message
                    - 0.55 * neighborhood_mean
                    + role_bias
                )
                target = self._sigmoid(preactivation)
                updated[vertex] = (
                    (1.0 - self.step_size) * state.get(vertex, 0.0)
                    + self.step_size * target
                )
            final_delta = max(
                (abs(updated[vertex] - state.get(vertex, 0.0)) for vertex in updated),
                default=0.0,
            )
            state = updated
            steps = step + 1
            if final_delta <= self.convergence_tol:
                converged = True
                break

        # Save local diagnostics on the transient candidate dictionaries. They
        # are consumed by readout and never reach ThreadStore.
        positive_edges = 0
        total_edges = 0
        for vertex, item in candidate_by_vertex.items():
            neighbors = adjacency.get(vertex) or {}
            positive_support = 0.0
            inhibition = 0.0
            context_support = 0.0
            bridge_support = 0.0
            normalizer = 0.0
            for neighbor, weight in sorted(neighbors.items()):
                compatibility = self._pair_compatibility(
                    vertex,
                    neighbor,
                    weight,
                    candidate_by_vertex,
                    affinity,
                )
                normalizer += abs(compatibility)
                total_edges += 1
                if compatibility >= 0.0:
                    positive_edges += 1
                    contribution = compatibility * state.get(neighbor, 0.0)
                    positive_support += contribution
                    neighbor_item = candidate_by_vertex.get(neighbor)
                    if neighbor.startswith("h:"):
                        bridge_support += 0.75 * contribution
                    elif neighbor_item and neighbor_item["thread_id"] == item["thread_id"]:
                        context_support += contribution
                    else:
                        bridge_support += contribution
                else:
                    inhibition += (-compatibility) * state.get(neighbor, 0.0)
            if normalizer > 1e-9:
                positive_support /= normalizer
                inhibition /= normalizer
                context_support /= normalizer
                bridge_support /= normalizer
            item["_seed_signal"] = self._clamp(vertex_seed.get(vertex, 0.0))
            item["_support_influence"] = self._clamp(positive_support)
            item["_inhibition_score"] = self._clamp(inhibition)
            # Re-evaluate the channels on the converged state.  This is the
            # role vector consumed by readout, and it matches the channels
            # that participated in every preceding update.
            final_messages = []
            for neighbor, weight in sorted(neighbors.items()):
                final_messages.append(
                    (
                        neighbor,
                        self._pair_compatibility(
                            vertex,
                            neighbor,
                            weight,
                            candidate_by_vertex,
                            affinity,
                        ),
                    )
                )
            channels = role_channels(vertex, final_messages, state)
            role_state[vertex] = channels
            item["_context_support"] = channels["local_support"]
            item["_bridge_support"] = channels["relay_mediation"]
            item["_query_drive"] = channels["query_drive"]
            item["_local_support"] = channels["local_support"]
            item["_relay_mediation"] = channels["relay_mediation"]
            item["_role_vector"] = dict(channels)
            item["_state_stability"] = self._clamp(1.0 - final_delta)

        # This is an energy-like trajectory diagnostic. It is deliberately
        # non-negative so legacy metric readers remain well-defined.
        activation_energy = 0.0
        for vertex, value in state.items():
            activation_energy += 0.5 * (
                float(value) - float(vertex_seed.get(vertex, 0.0))
            ) ** 2
        for vertex, neighbors in adjacency.items():
            for neighbor, weight in neighbors.items():
                if vertex >= neighbor:
                    continue
                compatibility = self._pair_compatibility(
                    vertex,
                    neighbor,
                    weight,
                    candidate_by_vertex,
                    affinity,
                )
                activation_energy += 0.05 * abs(compatibility) * state.get(vertex, 0.0) * state.get(neighbor, 0.0)

        node_state = {
            item["node_id"]: self._clamp(
                state.get(self._node_vertex(item["node_id"]), 0.0)
            )
            for item in candidates
        }
        # At least one positive interaction is a useful compatibility alias for
        # old diagnostics; the active algorithm itself uses ``edge_visits``.
        interaction_count = max(1, positive_edges) if candidates else 0
        support_density = (
            positive_edges / max(1, total_edges) if total_edges else 0.0
        )
        for item in candidates:
            item["_support_density"] = support_density
        return (
            node_state,
            steps,
            edge_visits,
            float(activation_energy),
            converged,
            float(final_delta if math.isfinite(final_delta) else 0.0),
        )

    @staticmethod
    def _diagnostic_role(
        item: Dict[str, Any],
        candidates: Sequence[Dict[str, Any]],
        question_type: str = "general",
    ) -> Tuple[str, Dict[str, float]]:
        """Project the causal role channels to compatibility labels.

        The vector has already been produced by the recurrent transition.  No
        question-type or rank heuristic is allowed to alter it here; this
        keeps the human-readable label from becoming a hidden selector.
        """
        vector = item.get("_role_vector") or {
            "query_drive": item.get("_query_drive", 0.0),
            "local_support": item.get("_local_support", item.get("_context_support", 0.0)),
            "relay_mediation": item.get("_relay_mediation", item.get("_bridge_support", 0.0)),
        }
        query_drive = max(0.0, min(1.0, float(vector.get("query_drive", 0.0))))
        local_support = max(0.0, min(1.0, float(vector.get("local_support", 0.0))))
        relay_mediation = max(0.0, min(1.0, float(vector.get("relay_mediation", 0.0))))
        item["_query_drive"] = query_drive
        item["_local_support"] = local_support
        item["_relay_mediation"] = relay_mediation
        item["_role_vector"] = {
            "query_drive": query_drive,
            "local_support": local_support,
            "relay_mediation": relay_mediation,
        }
        components = {
            "anchor": query_drive,
            "context": local_support,
            "bridge": relay_mediation,
        }
        priority = {"anchor": 2, "bridge": 1, "context": 0}
        role = max(components, key=lambda key: (components[key], priority[key]))
        return role, components

    def _readout(
        self,
        candidates: Sequence[Dict[str, Any]],
        question_type: str = "general",
    ) -> Tuple[List[Dict[str, Any]], float, float]:
        """Read the field with a deterministic submodular-style capacity rule.

        The recurrent field remains dense internally.  This final readout is a
        separate interface decision. The canonical query-similarity branch
        selects at most K source turns by their direct query affinity; the
        field branch remains a controlled ablation. Neither branch changes
        the persistent substrate.
        """
        if not candidates:
            return [], 0.0, self.visibility_floor
        activations = [self._clamp(item.get("activation_score", 0.0)) for item in candidates]
        if self.readout_mode == "query_similarity":
            readout_scores = [
                self._clamp(item.get("query_affinity", 0.0))
                for item in candidates
            ]
        else:
            readout_scores = [
                # Give the query-local graph field and direct query affinity
                # equal weight. Both terms are in [0, 1], so averaging keeps
                # the readout score in the same range without saturation.
                0.5 * activation
                + 0.5 * self._clamp(item.get("query_affinity", 0.0))
                for item, activation in zip(candidates, activations)
            ]
        mean = sum(readout_scores) / max(1, len(readout_scores))
        variance = sum((value - mean) ** 2 for value in readout_scores) / max(1, len(readout_scores))
        threshold = (
            0.0
            if self.readout_mode == "query_similarity"
            else max(
                self.visibility_floor,
                min(0.92, mean + 0.35 * math.sqrt(max(0.0, variance))),
            )
        )
        for item, score in zip(candidates, readout_scores):
            item["_readout_score"] = self._clamp(score)

        # Bind the transient role channels before readout so role diversity is
        # a causal feature of the utility, rather than a post-hoc label.
        for item in candidates:
            role, components = self._diagnostic_role(item, candidates, question_type)
            item["_readout_role"] = role
            item["_readout_components"] = components
            source_ids = tuple(str(value) for value in (item.get("source_turn_ids") or []))
            item["_readout_sources"] = source_ids or (str(item.get("node_id", "")),)
            # A conservative cost proxy is used only when an explicit readout
            # token budget is configured. Exact prompt cost is still measured
            # by the evaluator after serialization.
            item["_readout_cost"] = max(
                8,
                int(math.ceil(
                    (len(str(item.get("text", "")))
                     + 2 * len(item["_readout_sources"])
                     + len(str(item.get("thread_id", "")))
                     + 12) / 3.2
                )),
            )

        ordered_candidates = sorted(
            candidates,
            key=lambda item: (
                -float(item.get("_readout_score", 0.0)),
                -float(item.get("relay_mediation", item.get("_relay_mediation", 0.0))),
                int(item.get("thread_rank", 0)),
                int(item.get("ordinal", 0)),
                str(item.get("node_id", "")),
            ),
        )
        # K is the paper's source-turn readout budget and is independent of
        # question type. Every query exposes at most K source turns. The
        # compatibility attribute is still named max_readout_cells.
        capacity = self.max_readout_cells
        if self._legacy_item_guard is not None:
            capacity = min(capacity, self._legacy_item_guard)
        capacity = max(self.min_readout, capacity)
        token_budget = self.readout_token_budget
        selected: List[Dict[str, Any]] = []
        selected_ids: set[str] = set()
        covered_sources: set[str] = set()
        covered_threads: set[str] = set()
        covered_roles: set[str] = set()
        used_tokens = 0

        def cosine_redundancy(item: Dict[str, Any]) -> float:
            embedding = item.get("node_embedding") or []
            if not embedding or not selected:
                return 0.0
            return max(
                max(0.0, self._cosine(embedding, other.get("node_embedding") or []))
                for other in selected
            )

        def marginal_utility(item: Dict[str, Any]) -> float:
            score = float(item.get("_readout_score", 0.0))
            new_sources = len(
                set(item.get("_readout_sources") or ()) - covered_sources
            )
            new_thread = str(item.get("thread_id", "")) not in covered_threads
            role = str(item.get("_readout_role", "context"))
            new_role = role not in covered_roles
            relay = self._clamp(item.get("relay_mediation", item.get("_relay_mediation", 0.0)))
            local = self._clamp(item.get("local_support", item.get("_local_support", 0.0)))
            source_gain = min(1.0, new_sources / max(1, len(item.get("_readout_sources") or ())))
            structural_gain = 0.55 * relay + 0.45 * local
            # Diminishing returns on already represented threads/roles makes
            # the rule prefer complementary evidence without forcing a quota.
            coverage_gain = (
                0.74 * score
                + 0.13 * source_gain
                + 0.04 * float(new_thread)
                + 0.03 * float(new_role)
                + 0.06 * structural_gain
            )
            redundancy = cosine_redundancy(item)
            adjusted = coverage_gain * (1.0 - 0.18 * redundancy)
            cost = float(item.get("_readout_cost", 1.0))
            if token_budget:
                adjusted /= max(1.0, math.sqrt(cost))
            return max(0.0, adjusted)

        while ordered_candidates and len(selected) < capacity:
            feasible = [
                item for item in ordered_candidates
                if item["node_id"] not in selected_ids
                and (
                    self.readout_mode == "query_similarity"
                    or not token_budget
                    or used_tokens + int(item["_readout_cost"]) <= token_budget
                    or not selected
                )
            ]
            if not feasible:
                break
            if self.readout_mode == "query_similarity":
                best = min(
                    feasible,
                    key=lambda item: (
                        -float(item.get("_readout_score", 0.0)),
                        int(item.get("thread_rank", 0)),
                        int(item.get("ordinal", 0)),
                        str(item.get("node_id", "")),
                    ),
                )
                best["_marginal_utility"] = float(
                    best.get("_readout_score", 0.0)
                )
            else:
                best = max(
                    feasible,
                    key=lambda item: (
                        marginal_utility(item),
                        float(item.get("_readout_score", 0.0)),
                        -int(item.get("thread_rank", 0)),
                        -int(item.get("ordinal", 0)),
                        str(item.get("node_id", "")),
                    ),
                )
                best["_marginal_utility"] = marginal_utility(best)
            selected.append(best)
            selected_ids.add(str(best.get("node_id", "")))
            covered_sources.update(best.get("_readout_sources") or ())
            covered_threads.add(str(best.get("thread_id", "")))
            covered_roles.add(str(best.get("_readout_role", "context")))
            used_tokens += int(best.get("_readout_cost", 0))
            ordered_candidates = [item for item in ordered_candidates if item is not best]

        if len(selected) < self.min_readout:
            fallback = [item for item in sorted(
                candidates,
                key=lambda item: (
                    -float(item.get("_readout_score", 0.0)),
                    int(item.get("thread_rank", 0)),
                    int(item.get("ordinal", 0)),
                ),
            ) if str(item.get("node_id", "")) not in selected_ids]
            for item in fallback:
                selected.append(item)
                selected_ids.add(str(item.get("node_id", "")))
                if len(selected) >= min(self.min_readout, capacity):
                    break

        # The score determines membership; presentation follows original
        # source order as specified by the method.
        ordered_pairs = self._presentation_order(
            selected,
            [float(item.get("_readout_score", 0.0)) for item in selected],
            question_type,
        )
        selected = [item for item, _score in ordered_pairs]

        total_activation = sum(activations) or 1.0
        represented_activation = sum(
            self._clamp(item.get("activation_score", 0.0)) for item in selected
        )
        readout_coverage = min(1.0, represented_activation / total_activation)
        for item in selected:
            role = str(item.get("_readout_role", "context"))
            components = item.get("_readout_components") or {}
            item["role"] = role
            item["role_components"] = components
            item["role_gain"] = sum(components.values())
        return selected, readout_coverage, threshold

    def _local_affinity_matrix(
        self,
        candidates: Sequence[Dict[str, Any]],
        expanded: Sequence[ThreadCandidate],
        question_type: str = "general",
    ) -> List[List[float]]:
        """Expose fixed-substrate affinities for diagnostics and tests.

        This helper mirrors the local links used by the recurrent operator.
        It is deliberately a read-only matrix view; the active path does not
        optimize over subsets or solve a second graph problem.
        """
        size = len(candidates)
        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        for index in range(size):
            matrix[index][index] = 1.0

        link_weight: Dict[Tuple[str, str], float] = {}
        for item in expanded:
            for link in item.thread.links:
                key = (item.thread.thread_id, link.target_thread_id)
                link_weight[key] = max(link_weight.get(key, 0.0), float(link.weight))

        temporal = "temporal" in str(question_type).lower()
        for left in range(size):
            for right in range(left + 1, size):
                left_item = candidates[left]
                right_item = candidates[right]
                observable = max(
                    0.0,
                    self._cosine(
                        left_item.get("node_embedding") or [],
                        right_item.get("node_embedding") or [],
                    ),
                )
                if temporal:
                    lexical = max(
                        self._temporal_signal(left_item.get("text", "")),
                        self._temporal_signal(right_item.get("text", "")),
                    )
                    observable = max(observable, 0.35 * lexical)
                # Chronology links remain observable even when two source
                # cells have orthogonal embeddings.
                gate = 0.25 + 0.75 * observable
                if left_item["thread_id"] == right_item["thread_id"]:
                    gap = abs(int(left_item.get("ordinal", 0)) - int(right_item.get("ordinal", 0)))
                    weight = (0.35 ** max(1, gap)) * gate
                else:
                    link = max(
                        link_weight.get((left_item["thread_id"], right_item["thread_id"]), 0.0),
                        link_weight.get((right_item["thread_id"], left_item["thread_id"]), 0.0),
                    )
                    weight = 0.45 * gate * link
                if weight > 0.0:
                    matrix[left][right] = weight
                    matrix[right][left] = weight
        return matrix

    # Historical evaluator hook.  Keep the name as a thin read-only alias so
    # old smoke tests remain valid while the implementation has one pathway.
    def _kernel(
        self,
        candidates: Sequence[Dict[str, Any]],
        expanded: Sequence[ThreadCandidate],
        question_type: str = "general",
    ) -> List[List[float]]:
        return self._local_affinity_matrix(candidates, expanded, question_type)

    def _relations(
        self,
        evidence: Sequence[WorkspaceEvidence],
        expanded: Sequence[ThreadCandidate],
    ) -> List[WorkspaceRelation]:
        relations: List[WorkspaceRelation] = []
        by_thread_link = {
            (item.thread.thread_id, link.target_thread_id): link.relation
            for item in expanded
            for link in item.thread.links
        }
        for left_index, left in enumerate(evidence):
            for right in evidence[left_index + 1 :]:
                relation_type: Optional[str] = None
                if left.thread_id == right.thread_id and abs(left.ordinal - right.ordinal) == 1:
                    relation_type = "thread-continuation"
                elif (left.thread_id, right.thread_id) in by_thread_link:
                    relation_type = by_thread_link[(left.thread_id, right.thread_id)]
                elif (right.thread_id, left.thread_id) in by_thread_link:
                    relation_type = by_thread_link[(right.thread_id, left.thread_id)]
                if relation_type:
                    relations.append(
                        WorkspaceRelation(
                            source_evidence_id=left.evidence_id,
                            target_evidence_id=right.evidence_id,
                            relation_type=relation_type,
                        )
                    )
                if len(relations) >= self.max_relations:
                    return relations
        return relations


def build_transient_assembly(
    query: str,
    address: str,
    query_vector: Sequence[float],
    routes: Sequence[Any],
    expanded: Sequence[ThreadCandidate],
    plan: Optional[Dict[str, Any]] = None,
    *,
    item_guard: Optional[int] = None,
    readout_limit: Optional[int] = None,
    min_readout: int = 1,
    max_relations: int = 12,
    node_seed_weight: float = 0.80,
    rollout_steps: int = 20,
    step_size: float = 0.35,
    convergence_tol: float = 1e-3,
    readout_floor: float = 0.30,
    visibility_floor: Optional[float] = None,
    prompt_guard_tokens: int = 0,
    max_readout_cells: Optional[int] = 40,
    readout_token_budget: int = 0,
    readout_mode: str = "field",
    compatibility_scale: float = 1.35,
    inhibition_scale: float = 0.35,
    **legacy_options: Any,
) -> MemoryWorkspace:
    if item_guard is None:
        item_guard = readout_limit
    if item_guard is None and "max_evidence" in legacy_options:
        item_guard = legacy_options["max_evidence"]
    if "min_evidence" in legacy_options:
        min_readout = legacy_options["min_evidence"]
    return TransientAssemblyBuilder(
        item_guard=item_guard,
        min_readout=min_readout,
        max_relations=max_relations,
        node_seed_weight=node_seed_weight,
        rollout_steps=rollout_steps,
        step_size=step_size,
        convergence_tol=convergence_tol,
        readout_floor=readout_floor,
        visibility_floor=visibility_floor,
        prompt_guard_tokens=prompt_guard_tokens,
        max_readout_cells=max_readout_cells,
        readout_token_budget=readout_token_budget,
        readout_mode=readout_mode,
        compatibility_scale=compatibility_scale,
        inhibition_scale=inhibition_scale,
        **legacy_options,
    ).build(query, address, query_vector, routes, expanded, plan)


def build_thread_workspace(
    query: str,
    address: str,
    query_vector: Sequence[float],
    routes: Sequence[Any],
    expanded: Sequence[ThreadCandidate],
    plan: Optional[Dict[str, Any]] = None,
    **options: Any,
) -> MemoryWorkspace:
    """Compatibility entry point for existing benchmark scripts."""
    return build_transient_assembly(
        query,
        address,
        query_vector,
        routes,
        expanded,
        plan,
        **options,
    )


# Canonical entry point for new experiments.
build_transient_role_state = build_transient_assembly


# Compatibility aliases.  New code should use the descriptive names above.
AssemblyCell = WorkspaceEvidence
AssemblyRelation = WorkspaceRelation
TransientAssembly = MemoryWorkspace
TransientAssemblyState = MemoryWorkspace
AssemblyInference = TransientRoleBindingBuilder
ThreadWorkspaceBuilder = TransientRoleBindingBuilder
TransientAssemblyBuilder = TransientRoleBindingBuilder
TransientAssemblyEngine = TransientRoleBindingBuilder

# Canonical conceptual names for the current method.  The implementation is
# intentionally shared with the historical aliases so benchmark entry points
# cannot silently exercise a different algorithm.
TransientRoleBindingState = MemoryWorkspace
TransientRoleBindingCell = WorkspaceEvidence
