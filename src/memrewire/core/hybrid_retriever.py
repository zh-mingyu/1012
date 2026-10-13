"""One-shot addressing followed by transient local assembly."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from memrewire.core.models.memory_entry import MemoryEntry
from memrewire.core.settings import settings as config
from memrewire.core.cell_graph_store import CellGraphStore
if TYPE_CHECKING:
    from memrewire.core.utils.llm_client import LLMClient
from memrewire.core.query_context import QueryContext, build_query_context


class MemRewireRetriever:
    """Route once to source cells, then run the shared local state update.

    The class name remains for API compatibility. The active path has no
    three-channel candidate union, no requirement-specific retrieval, and no
    query-time mutation of the persistent substrate.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: CellGraphStore,
        enable_planning: Optional[bool] = None,
        enable_reflection: bool = False,
        max_reflection_rounds: int = 0,
        enable_parallel_retrieval: bool = True,
        max_retrieval_workers: int = 1,
    ) -> None:
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.address_route_width = int(
            getattr(config, "ADDRESS_ROUTE_WIDTH", getattr(config, "CELL_ROUTE_TOP_K", 16))
        )
        # The canonical method scores every cell once and uses a
        # mean-plus-standard-deviation threshold to form the seed set.
        self.cell_route_probe_top_k = 0
        self.receptive_field_hops = int(
            getattr(config, "RECEPTIVE_FIELD_HOPS", getattr(config, "CELL_MAX_HOPS", 3))
        )
        self.receptive_field_scope = int(
            getattr(config, "RECEPTIVE_FIELD_SCOPE", getattr(config, "CELL_MAX_EXPANDED", 32))
        )
        self.broad_receptive_field_scope = self.receptive_field_scope
        # Historical attributes remain aliases for external launchers.
        self.cell_route_top_k = self.address_route_width
        self.cell_max_hops = self.receptive_field_hops
        self.cell_max_expanded = self.receptive_field_scope
        self.cell_adaptive_max_expanded = self.broad_receptive_field_scope
        # The field determines its own visible basins. This guard is applied
        # only when serializing a prompt for the answer model.
        self.query_context_prompt_guard_tokens = int(
            getattr(config, "CONTEXT_PROMPT_GUARD_TOKENS", 6000)
        )
        self.source_readout_k = int(
            getattr(
                config,
                "SOURCE_READOUT_K",
                getattr(config, "CONTEXT_MAX_READOUT_CELLS", 4),
            )
        )
        # Historical field name retained for evaluator compatibility.
        self.query_context_max_readout_cells = self.source_readout_k
        self.query_context_readout_token_budget = int(
            getattr(config, "CONTEXT_READOUT_TOKEN_BUDGET", 0)
        )
        self.query_context_readout_mode = str(
            getattr(config, "CONTEXT_READOUT_MODE", "query_similarity")
        )
        self.query_context_min_visible_cells = int(
            getattr(config, "CONTEXT_MIN_VISIBLE_CELLS", 1)
        )
        self.query_context_max_relations = int(getattr(config, "CONTEXT_MAX_RELATIONS", 12))
        self.query_context_node_seed_weight = float(
            getattr(config, "CONTEXT_NODE_SEED_WEIGHT", 0.80)
        )
        self.query_context_rollout_steps = int(
            getattr(config, "CONTEXT_ROLLOUT_STEPS", 20)
        )
        self.query_context_step_size = float(
            getattr(config, "CONTEXT_STEP_SIZE", 0.35)
        )
        self.query_context_convergence_tol = float(
            getattr(config, "CONTEXT_CONVERGENCE_TOL", 1e-3)
        )
        self.query_context_visibility_floor = float(
            getattr(config, "CONTEXT_VISIBILITY_FLOOR", 0.30)
        )
        self.query_context_compatibility_scale = float(
            getattr(config, "CONTEXT_COMPATIBILITY_SCALE", 1.35)
        )
        self.query_context_inhibition_scale = float(
            getattr(config, "CONTEXT_INHIBITION_SCALE", 0.35)
        )
        self.enable_planning = (
            bool(enable_planning)
            if enable_planning is not None
            else bool(getattr(config, "ENABLE_CELL_ADDRESS_PLANNER", False))
        )
        # Reflection is intentionally not part of the main method: it would
        # turn local assembly back into repeated query-conditioned retrieval.
        self.enable_reflection = False

    def retrieve_query_context(
        self,
        query: str,
        enable_reflection: Optional[bool] = None,
    ) -> QueryContext:
        logger = logging.getLogger("paired_graph.retrieval")
        plan = self._plan_address(query) if self.enable_planning else {}
        address = str(plan.get("address") or query).strip() or query
        route_started = time.perf_counter()
        logger.info("RETRIEVAL_ROUTE_START cells=%s seed_rule=mean-plus-std", self.vector_store.count())
        probed_routes, query_vector = self.vector_store.route_seeds(address)
        logger.info(
            "RETRIEVAL_ROUTE_DONE seed_cells=%s elapsed_seconds=%.2f",
            len(probed_routes),
            time.perf_counter() - route_started,
        )
        routes = probed_routes
        # The paper defines only a hop limit. It does not add a separate
        # query-time cell truncation scope, so expansion visits every distinct
        # cell reachable within H hops.
        expansion_started = time.perf_counter()
        logger.info(
            "RETRIEVAL_EXPANSION_START seeds=%s max_hops=%s max_cells=unbounded",
            len(routes),
            self.receptive_field_hops,
        )
        expanded = self.vector_store.expand(
            routes,
            max_hops=self.receptive_field_hops,
            max_cells=None,
        )
        logger.info(
            "RETRIEVAL_EXPANSION_DONE visited_cells=%s traversed_edges=%s elapsed_seconds=%.2f",
            len(expanded),
            self.vector_store.last_expansion_edges,
            time.perf_counter() - expansion_started,
        )
        assembly_started = time.perf_counter()
        logger.info("RETRIEVAL_ASSEMBLY_START expanded_cells=%s", len(expanded))
        query_context = build_query_context(
            query=query,
            address=address,
            query_vector=query_vector,
            routes=routes,
            expanded=expanded,
            plan=plan,
            min_readout=self.query_context_min_visible_cells,
            max_relations=self.query_context_max_relations,
            node_seed_weight=self.query_context_node_seed_weight,
            rollout_steps=self.query_context_rollout_steps,
            step_size=self.query_context_step_size,
            convergence_tol=self.query_context_convergence_tol,
            visibility_floor=self.query_context_visibility_floor,
            prompt_guard_tokens=self.query_context_prompt_guard_tokens,
            max_readout_cells=self.source_readout_k,
            readout_token_budget=self.query_context_readout_token_budget,
            readout_mode=self.query_context_readout_mode,
            compatibility_scale=self.query_context_compatibility_scale,
            inhibition_scale=self.query_context_inhibition_scale,
        )
        logger.info(
            "RETRIEVAL_ASSEMBLY_DONE candidate_nodes=%s evidence_nodes=%s elapsed_seconds=%.2f",
            query_context.node_candidate_count,
            len(query_context.evidence),
            time.perf_counter() - assembly_started,
        )
        query_context.address_planner_calls = 1 if self.enable_planning else 0
        query_context.indexed_cell_count = self.vector_store.count()
        graph_diagnostics = getattr(self.vector_store, "last_graph_diagnostics", {}) or {}
        query_context.topology_mode = str(graph_diagnostics.get("mode", "operator-aware"))
        query_context.topology_candidate_edges = int(graph_diagnostics.get("candidate_edges", 0))
        query_context.topology_selected_edges = int(
            graph_diagnostics.get("selected_edges", graph_diagnostics.get("selected_optional_edges", 0))
        )
        query_context.topology_optional_edges = int(graph_diagnostics.get("selected_optional_edges", 0))
        query_context.topology_mandatory_edges = int(graph_diagnostics.get("mandatory_edges", 0))
        query_context.topology_edge_budget = int(graph_diagnostics.get("baseline_edge_budget", 0))
        query_context.topology_operator_objective = float(graph_diagnostics.get("objective", 0.0))
        query_context.topology_mean_gain = float(graph_diagnostics.get("mean_optional_gain", 0.0))
        query_context.topology_cross_session_edges = int(graph_diagnostics.get("cross_session_edges", 0))
        query_context.local_edge_visits += self.vector_store.last_expansion_edges
        query_context.route_probe_count = self.vector_store.count()
        query_context.adaptive_route = False
        query_context.route_width = len(routes)
        query_context.receptive_field_scope = len(expanded)
        print(
            f"[MemRewire] routed={len(routes)} expanded={len(expanded)} "
            f"nodes={query_context.node_candidate_count} visible={len(query_context.evidence)} "
            f"activation={query_context.activation_total:.3f}"
        )
        return query_context

    def retrieve(self, query: str, enable_reflection: Optional[bool] = None) -> List[MemoryEntry]:
        """Compatibility API returning the selected query_context nodes as flat items."""
        query_context = self.retrieve_query_context(query, enable_reflection=enable_reflection)
        return [
            MemoryEntry(
                entry_id=item.node_id,
                lossless_restatement=item.text,
                timestamp=item.timestamp,
                source_turn_ids=item.source_turn_ids,
                cell_id=item.cell_id,
            )
            for item in query_context.evidence
        ]
    def _plan_address(self, query: str) -> Dict[str, Any]:
        """Create one address plus rendering diagnostics; never emit search queries."""
        prompt = f"""
Read the question and create one address for a source-cell memory substrate.

Question: {query}

The address should identify the answer target, its event or relation, and any
        temporal/order relation. Do not produce keyword lists, multiple queries,
retrieval channels, requirements, or candidate memories.

Return ONLY JSON:
{{
  "address": "one concise event/relation address",
  "question_type": "single-hop|multi-hop|temporal|open-domain|adversarial|general",
  "complexity": "LOW|MEDIUM|HIGH",
  "temporal_direction": "forward|backward|both"
}}
"""
        messages = [
            {"role": "system", "content": "You create one precise memory address and must return valid JSON."},
            {"role": "user", "content": prompt},
        ]
        try:
            response_format = {"type": "json_object"} if getattr(config, "USE_JSON_FORMAT", False) else None
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.0,
                response_format=response_format,
                stage="retrieval.cell_address",
            )
            plan = self.llm_client.extract_json(response)
            if not isinstance(plan, dict):
                raise ValueError("cell address planner must return an object")
            return plan
        except Exception as error:
            print(f"[MemRewire] address planning failed; using question: {error}")
            return {
                "address": query,
                "question_type": "general",
                "complexity": "LOW",
                "temporal_direction": "both",
            }


QueryContextRetriever = MemRewireRetriever
