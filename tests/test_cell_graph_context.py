"""Dependency-light tests for the active cell query_context path."""

import json
import hashlib
import tempfile

import numpy as np

from memrewire.core.memory_builder import MemoryBuilder
from memrewire.core.models.memory_entry import Dialogue, MemoryCell, SourceTurn
from memrewire.core.cell_graph_store import CellGraphStore
from memrewire.core.hybrid_retriever import QueryContextRetriever
from memrewire.core.query_context import (
    QueryContext,
    QueryContextBuilder,
    ContextEvidence,
    ContextRelation,
    build_cell_query_context,
)
from memrewire.core.answer_generator import AnswerGenerator
from memrewire.core.settings import settings as config


class FakeEmbedding:
    dimension = 4

    def encode_documents(self, texts):
        return np.asarray(
            [
                [float("road" in text.lower()), float("accident" in text.lower()),
                 float("future" in text.lower()), 1.0]
                for text in texts
            ],
            dtype="float32",
        )

    def encode_single(self, text, is_query=False):
        text = text.lower()
        return np.asarray(
            [float("road" in text), float("accident" in text), float("future" in text), 1.0],
            dtype="float32",
        )


class FakeLLM:
    def chat_completion(self, *_args, **_kwargs):
        return json.dumps({
            "cell_head": "Caroline support group and education",
            "nodes": [{
                "text": "Caroline attended an LGBTQ support group.",
                "source_turn_ids": ["D1:3"],
                "ordinal": 0,
            }],
        })

    @staticmethod
    def extract_json(response):
        return json.loads(response)


class BundleLLM:
    def chat_completion(self, *_args, **_kwargs):
        return json.dumps({
            "cells": [
                {
                    "address_synopsis": "support group disclosure",
                    "source_turn_ids": ["D2:1", "D2:2"],
                    "nodes": [{
                        "text": "Caroline attended a support group.",
                        "source_turn_ids": ["D2:1"],
                        "ordinal": 0,
                    }],
                },
                {
                    "address_synopsis": "education plan",
                    "source_turn_ids": ["D2:3", "D2:4"],
                    "nodes": [{
                        "text": "Caroline plans further education.",
                        "source_turn_ids": ["D2:3"],
                        "ordinal": 0,
                    }],
                },
            ]
        })

    @staticmethod
    def extract_json(response):
        return json.loads(response)


class NoopPlannerLLM:
    """The default path must not call the retrieval-time planner."""

    def chat_completion(self, *_args, **_kwargs):
        raise AssertionError("retrieval-time planning should be disabled")

    @staticmethod
    def extract_json(response):
        return json.loads(response)


def _cells():
    return [
        MemoryCell(
            cell_id="t1",
            head="Melanie road trip and accident",
            session_id="D1",
            ordinal=0,
            nodes=[
                SourceTurn(
                    node_id="t1:n1",
                    text="Melanie took a road trip with her son.",
                    source_turn_ids=["D1:1"],
                    ordinal=0,
                    timestamp="2023-06-10",
                    speakers=["Melanie"],
                ),
                SourceTurn(
                    node_id="t1:n2",
                    text="Melanie's son was injured in a car accident.",
                    source_turn_ids=["D1:2"],
                    ordinal=1,
                    timestamp="2023-06-12",
                    speakers=["Melanie"],
                ),
            ],
        ),
        MemoryCell(
            cell_id="t2",
            head="Melanie future road trip",
            session_id="D2",
            ordinal=1,
            nodes=[SourceTurn(
                node_id="t2:n1",
                text="Melanie would consider another road trip.",
                source_turn_ids=["D2:1"],
                ordinal=0,
                speakers=["Melanie"],
            )],
        ),
    ]


def test_cell_builder_preserves_missing_turn_provenance():
    with tempfile.TemporaryDirectory() as directory:
        store = CellGraphStore(directory, embedding_model=FakeEmbedding(), table_name="cells")
        builder = MemoryBuilder(FakeLLM(), store, window_size=20)
        builder.add_dialogues([
            Dialogue(dialogue_id=1, speaker="Caroline", content="group", source_id="D1:3", session_id="D1"),
            Dialogue(dialogue_id=2, speaker="Caroline", content="education", source_id="D1:4", session_id="D1"),
        ])
        builder.process_remaining()
        cell = store.get_all_cells()[0]
        nodes = cell.nodes
        assert {source for node in nodes for source in node.source_turn_ids} == {"D1:3", "D1:4"}
        assert nodes[0].text == "Caroline: group"
        assert nodes[1].text == "Caroline: education"
        assert not hasattr(cell, "continuity_keys")
        assert all(not hasattr(node, "continuity_keys") for node in nodes)


def test_event_bundle_creates_multiple_contiguous_cells(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        store = CellGraphStore(directory, embedding_model=FakeEmbedding(), table_name="cells")
        monkeypatch.setattr(config, "ENABLE_CELL_SEGMENTATION", True, raising=False)
        builder = MemoryBuilder(BundleLLM(), store, window_size=20)
        builder.add_dialogues([
            Dialogue(dialogue_id=1, speaker="Caroline", content="group", source_id="D2:1", session_id="D2"),
            Dialogue(dialogue_id=2, speaker="Caroline", content="support", source_id="D2:2", session_id="D2"),
            Dialogue(dialogue_id=3, speaker="Caroline", content="school", source_id="D2:3", session_id="D2"),
            Dialogue(dialogue_id=4, speaker="Caroline", content="course", source_id="D2:4", session_id="D2"),
        ])
        builder.process_remaining()
        cells = store.get_all_cells()
        assert len(cells) == 2
        assert [cell.session_id for cell in cells] == ["D2", "D2"]
        assert [
            source_id
            for cell in cells
            for node in cell.nodes
            for source_id in node.source_turn_ids
        ] == ["D2:1", "D2:2", "D2:3", "D2:4"]
        assert cells[0].links[0].relation == "chronology-next"


def test_local_query_context_selects_connected_multi_evidence():
    with tempfile.TemporaryDirectory() as directory:
        store = CellGraphStore(directory, embedding_model=FakeEmbedding(), table_name="cells")
        store.add_cells(_cells())
        routes, vector = store.route("accident", top_k=1)
        expanded = store.expand(routes, max_hops=2, max_cells=4)
        query_context = build_cell_query_context(
            "Why might Melanie take another road trip after the accident?",
            "another road trip after accident",
            vector,
            routes,
            expanded,
            {
                "question_type": "multi-hop",
                "complexity": "HIGH",
            },
            max_evidence=4,
        )
        assert len(query_context.evidence) >= 2
        assert {source for item in query_context.evidence for source in item.source_turn_ids} >= {"D1:2", "D2:1"}
        assert query_context.sufficient
        assert query_context.coverage_mode == "transient-assembly"
        assert query_context.activation_iterations > 0
        assert query_context.transport_flow_units > 0
        assert query_context.transport_cost >= 0.0
        assert query_context.graph_edge_count > 0
        assert query_context.global_route_calls == 1
        assert not hasattr(query_context, "requirements")
        assert not hasattr(query_context.evidence[0], "requirement_ids")
        assert query_context.selection_budget == 4
        assert query_context.selection_rounds == len(query_context.evidence)


def test_expansion_diagnostics_and_candidate_stage_ids_are_serializable():
    with tempfile.TemporaryDirectory() as directory:
        store = CellGraphStore(directory, embedding_model=FakeEmbedding(), table_name="cells")
        store.add_cells(_cells())
        routes, vector = store.route("accident", top_k=1)
        expanded = store.expand(routes, max_hops=2, max_cells=4)

        assert len(store.last_expansion_by_hop) == 3
        assert set(store.last_expansion_by_hop[0]["cell_ids"]) == {
            route.cell.cell_id for route in routes
        }
        assert all(
            {
                "from_cell_id", "to_cell_id", "from_hop", "to_hop",
                "relation", "weight", "next_score", "accepted", "reason",
            }.issubset(edge)
            for edge in store.last_expansion_trace
        )
        json.dumps(store.last_expansion_trace)

        query_context = build_cell_query_context(
            "What happened in the accident?",
            "accident",
            vector,
            routes,
            expanded,
            {"question_type": "single-hop"},
            max_readout_cells=2,
            readout_mode="query_similarity",
        )
        assert query_context.candidate_cell_ids
        assert query_context.candidate_node_ids
        assert set(query_context.candidate_cell_ids) == {
            item.cell.cell_id for item in expanded
        }


def test_same_node_role_changes_with_query():
    with tempfile.TemporaryDirectory() as directory:
        store = CellGraphStore(directory, embedding_model=FakeEmbedding(), table_name="cells")
        store.add_cells(_cells())
        routes, vector = store.route("accident", top_k=1)
        expanded = store.expand(routes, max_hops=0, max_cells=1)
        direct = build_cell_query_context(
            "What happened to the son?", "son accident", vector, routes, expanded,
            {"question_type": "single-hop"},
            max_evidence=2,
        )
        temporal = build_cell_query_context(
            "When did the accident happen?", "time of accident", vector, routes, expanded,
            {"question_type": "temporal"},
            max_evidence=2,
        )
        direct_item = next(item for item in direct.evidence if item.node_id == "t1:n2")
        temporal_item = next(item for item in temporal.evidence if item.node_id == "t1:n2")
        assert direct_item.contribution != temporal_item.contribution
    assert temporal_item.contribution == "temporal-anchor"


def test_role_binding_is_task_time_and_decomposes_gain():
    with tempfile.TemporaryDirectory() as directory:
        store = CellGraphStore(directory, embedding_model=FakeEmbedding(), table_name="cells")
        store.add_cells(_cells())
        routes, vector = store.route("accident", top_k=1)
        expanded = store.expand(routes, max_hops=2, max_cells=4)
        query_context = build_cell_query_context(
            "Why might Melanie take another road trip after the accident?",
            "another road trip after accident",
            vector,
            routes,
            expanded,
            {"question_type": "multi-hop", "complexity": "HIGH"},
            max_evidence=4,
        )
        assert query_context.role_binding_mode == "field-coupled"
        assert query_context.evidence
        assert sum(query_context.role_counts.values()) == len(query_context.evidence)
        for item in query_context.evidence:
            assert item.role in {"anchor", "bridge", "context"}
            assert abs(sum(item.role_components.values()) - item.marginal_gain) < 2e-5
            assert set(item.role_vector) == {
                "query_drive", "local_support", "relay_mediation"
            }
        assert all(not hasattr(node, "role") for cell in store.get_all_cells() for node in cell.nodes)


def test_same_memory_node_binds_as_anchor_or_bridge_by_task():
    with tempfile.TemporaryDirectory() as directory:
        store = CellGraphStore(directory, embedding_model=FakeEmbedding(), table_name="cells")
        store.add_cells(_cells())
        direct_routes, direct_vector = store.route("future road trip", top_k=1)
        direct = build_cell_query_context(
            "What future road trip might Melanie take?",
            "future road trip",
            direct_vector,
            direct_routes,
            store.expand(direct_routes, max_hops=2, max_cells=4),
            {"question_type": "single-hop"},
            max_evidence=4,
        )
        multi_routes, multi_vector = store.route("another road trip after accident", top_k=1)
        multi = build_cell_query_context(
            "Why might Melanie take another road trip after the accident?",
            "another road trip after accident",
            multi_vector,
            multi_routes,
            store.expand(multi_routes, max_hops=2, max_cells=4),
            {"question_type": "multi-hop"},
            max_evidence=4,
        )
        direct_item = next(item for item in direct.evidence if item.node_id == "t2:n1")
        multi_item = next(item for item in multi.evidence if item.node_id == "t2:n1")
        assert any(
            abs(direct_item.role_vector[key] - multi_item.role_vector[key]) > 1e-6
            for key in ("query_drive", "local_support", "relay_mediation")
        )


def test_retriever_routes_once_and_assembles_locally():
    with tempfile.TemporaryDirectory() as directory:
        store = CellGraphStore(directory, embedding_model=FakeEmbedding(), table_name="cells")
        store.add_cells(_cells())
        retriever = QueryContextRetriever(
            NoopPlannerLLM(),
            store,
            enable_planning=False,
        )
        query_context = retriever.retrieve_query_context("What happened in the accident?")
        assert query_context.global_route_calls == 1
        assert query_context.address_planner_calls == 0
        assert query_context.evidence
        assert not hasattr(query_context, "requirements")
        assert all(not hasattr(item, "matched_terms") for item in query_context.evidence)


def test_default_readout_is_field_derived_without_a_cell_guard():
    """The active path derives visible cells from the field, not a fixed K."""
    with tempfile.TemporaryDirectory() as directory:
        store = CellGraphStore(directory, embedding_model=FakeEmbedding(), table_name="cells")
        store.add_cells(_cells())
        routes, vector = store.route("accident", top_k=1)
        expanded = store.expand(routes, max_hops=2, max_cells=4)
        unguarded = build_cell_query_context(
            "What happened after the accident?",
            "accident aftermath",
            vector,
            routes,
            expanded,
            {"question_type": "single-hop"},
        )
        compatibility_guard = build_cell_query_context(
            "What happened after the accident?",
            "accident aftermath",
            vector,
            routes,
            expanded,
            {"question_type": "single-hop"},
            readout_limit=1,
        )
        assert unguarded.readout_limit == 0
        assert unguarded.selection_budget == 0
        assert len(unguarded.evidence) >= len(compatibility_guard.evidence)
        assert unguarded.visible_node_ids == [item.node_id for item in unguarded.evidence]


def test_query_similarity_readout_skips_rollout_and_uses_direct_top_k():
    with tempfile.TemporaryDirectory() as directory:
        store = CellGraphStore(directory, embedding_model=FakeEmbedding(), table_name="cells")
        store.add_cells(_cells())
        routes, vector = store.route("accident", top_k=2)
        expanded = store.expand(routes, max_hops=2, max_cells=4)
        candidates = QueryContextBuilder(
            readout_mode="query_similarity"
        )._flatten(expanded, vector)
        expected = min(
            candidates,
            key=lambda item: (
                -item["query_affinity"],
                item["cell_rank"],
                item["ordinal"],
                item["node_id"],
            ),
        )["node_id"]
        query_context = build_cell_query_context(
            "What happened in the accident?",
            "accident",
            vector,
            routes,
            expanded,
            {"question_type": "single-hop"},
            max_readout_cells=1,
            readout_mode="query_similarity",
        )
        assert query_context.visible_node_ids == [expected]
        assert query_context.dynamics_mode == "not-run-query-similarity"
        assert query_context.activation_iterations == 0
        assert query_context.message_edge_visits == 0
        assert query_context.readout_rule == "query-similarity-top-k"
        assert query_context.sufficient


def test_shared_local_transition_is_permutation_equivariant():
    """Renaming vertices must only rename the resulting activation field."""
    builder = QueryContextBuilder(rollout_steps=8, convergence_tol=1e-8)
    candidates = [
        {"node_id": "a", "cell_id": "t", "node_embedding": [1.0, 0.0],
         "query_affinity": 0.9, "route_score": 0.8},
        {"node_id": "b", "cell_id": "t", "node_embedding": [0.8, 0.2],
         "query_affinity": 0.4, "route_score": 0.8},
        {"node_id": "c", "cell_id": "t", "node_embedding": [0.0, 1.0],
         "query_affinity": 0.1, "route_score": 0.8},
    ]
    adjacency = {
        "n:a": {"n:b": 1.0},
        "n:b": {"n:a": 1.0, "n:c": 0.7},
        "n:c": {"n:b": 0.7},
    }
    seed = {"n:a": 0.8, "n:b": 0.2, "n:c": 0.0}
    first, *_ = builder._recurrent_rollout(adjacency, seed, candidates)

    rename = {"a": "z", "b": "x", "c": "y"}
    permuted_candidates = [
        {**item, "node_id": rename[item["node_id"]]}
        for item in candidates
    ]
    permuted_adjacency = {
        f"n:{rename[left[2:]]}": {
            f"n:{rename[right[2:]]}": weight
            for right, weight in neighbors.items()
        }
        for left, neighbors in adjacency.items()
    }
    permuted_seed = {f"n:{rename[key[2:]]}": value for key, value in seed.items()}
    second, *_ = builder._recurrent_rollout(
        permuted_adjacency, permuted_seed, permuted_candidates
    )
    for old_id, new_id in rename.items():
        assert abs(first[old_id] - second[new_id]) < 1e-7


def test_prompt_serialization_drops_partial_relations(monkeypatch):
    """A relation is shown only when both of its source cells are visible."""
    monkeypatch.setattr(config, "CONTEXT_SOURCE_ONLY_SERIALIZATION", False, raising=False)
    query_context = QueryContext(
        query_context_id="assembly-test",
        query="Which event happened first?",
        prompt_guard_tokens=1,
        evidence=[
            ContextEvidence(
                evidence_id="W1", cell_id="t1", node_id="n1",
                text="The first event.",
            ),
            ContextEvidence(
                evidence_id="W2", cell_id="t1", node_id="n2",
                text="The second event.",
            ),
        ],
        relations=[ContextRelation(
            source_evidence_id="W1",
            target_evidence_id="W2",
            relation_type="cell-continuation",
        )],
    )
    rendered = AnswerGenerator._format_query_context(query_context)
    assert "[Assembly cell W1]" in rendered
    assert "W1 -> W2" not in rendered
    assert query_context.serialized_cell_count == 1


def test_lancedb_metadata_maps_are_lazy_and_reset_between_roots():
    """Expansion loads payload metadata once, and clear prevents cross-root reuse."""
    with tempfile.TemporaryDirectory() as directory:
        store = CellGraphStore(directory, embedding_model=FakeEmbedding(), table_name="cells")
        store.add_cells(_cells())
        assert store._payload_index is None
        routes, _vector = store.route("accident", top_k=1)
        store.expand(routes, max_hops=1, max_cells=2)
        if store._use_lancedb:
            assert store._payload_index is not None
        store.clear()
        assert store._payload_index is None
        assert store._head_index is None


def test_transient_assembly_is_query_local_and_persistent_store_is_unchanged():
    with tempfile.TemporaryDirectory() as directory:
        store = CellGraphStore(directory, embedding_model=FakeEmbedding(), table_name="cells")
        store.add_cells(_cells())
        before = hashlib.sha256(
            json.dumps(
                [cell.model_dump() for cell in store.get_all_cells()],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        routes, vector = store.route("accident", top_k=1)
        expanded = store.expand(routes, max_hops=2, max_cells=4)
        query_context = build_cell_query_context(
            "What happened after the accident?",
            "accident aftermath",
            vector,
            routes,
            expanded,
            {"question_type": "single-hop"},
            max_evidence=4,
        )
        after = hashlib.sha256(
            json.dumps(
                [cell.model_dump() for cell in store.get_all_cells()],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        assert before == after
        assert query_context.dynamics_mode == "local-recurrent-assembly"
        assert query_context.rollout_steps > 0
        assert query_context.readout_threshold > 0.0
        assert query_context.message_edge_visits > 0
        assert all(item.readout_score > 0.0 for item in query_context.evidence)


def test_legacy_mass_and_flow_arguments_do_not_change_assembly():
    with tempfile.TemporaryDirectory() as directory:
        store = CellGraphStore(directory, embedding_model=FakeEmbedding(), table_name="cells")
        store.add_cells(_cells())
        routes, vector = store.route("accident", top_k=1)
        expanded = store.expand(routes, max_hops=2, max_cells=4)
        common = dict(
            query="What happened after the accident?",
            address="accident aftermath",
            query_vector=vector,
            routes=routes,
            expanded=expanded,
            plan={"question_type": "single-hop"},
            max_evidence=4,
        )
        first = build_cell_query_context(**common, mass_threshold=0.2, flow_temperature=0.1)
        second = build_cell_query_context(**common, mass_threshold=0.95, flow_temperature=2.0)
        assert first.selected_node_ids == second.selected_node_ids
        assert [item.activation_score for item in first.evidence] == [
            item.activation_score for item in second.evidence
        ]


def test_locomo_calendar_hint_uses_session_anchor_without_iso_format():
    hint = AnswerGenerator._calendar_hints(
        "I painted that lake sunrise last year.",
        "1:56 pm on 8 May, 2023",
    )
    assert '"last year" -> 2022' in hint
    friday = AnswerGenerator._calendar_hints(
        "I went there last Friday.",
        "12:00 pm on 15 July, 2023",
    )
    assert "Friday before 15 July 2023" in friday


def test_graph_coverage_requires_content_observability():
    candidates = [
        {
            "cell_id": "t1", "node_id": "n1", "ordinal": 0,
            "node_embedding": [1.0, 0.0],
        },
        {
            "cell_id": "t1", "node_id": "n2", "ordinal": 1,
            "node_embedding": [0.0, 1.0],
        },
        {
            "cell_id": "t1", "node_id": "n3", "ordinal": 2,
            "node_embedding": [1.0, 0.0],
        },
    ]
    builder = build_cell_query_context.__globals__["QueryContextBuilder"]()
    kernel = builder._kernel(
        candidates,
        [],
        question_type="temporal",
    )
    # Temporal observability has a non-zero floor so lexical date turns are
    # not discarded solely because their embeddings are orthogonal.
    assert kernel[0][1] > 0.0
    assert kernel[0][2] > 0.0
    compositional = builder._kernel(
        candidates,
        [],
        question_type="set-valued",
    )
    assert compositional[0][1] > 0.0


def test_structured_answer_is_normalized_to_plain_text():
    assert AnswerGenerator._answer_text(["clarinet", "violin"], "") == "clarinet, violin"
    assert AnswerGenerator._normalise_count_answer(
        "3", "How many screenplays has Joanna written?"
    ) == "three"
    assert AnswerGenerator._normalise_count_answer(
        "2023", "When did Joanna write it?"
    ) == "2023"


def test_temporal_hint_preserves_relative_phrase_and_resolves_year():
    hint = AnswerGenerator._calendar_hints(
        "I joined the group the Friday before 15 July 2023.",
        "12:00 pm on 20 July, 2023",
    )
    assert "preserve phrase" in hint
    assert "14 July 2023" in hint
    assert AnswerGenerator._normalise_temporal_answer(
        "last year", 'calendar-hint="last year" -> 2022'
    ) == "2022"
    assert AnswerGenerator._normalise_temporal_answer(
        "14 July 2023",
        'calendar-hint="the Friday before 15 July 2023" -> preserve phrase; '
        'resolved date=14 July 2023',
    ) == "the Friday before 15 July 2023"
    month_hint = AnswerGenerator._calendar_hints(
        "Our group performs next month.",
        "10:00 am on 20 January, 2023",
    )
    assert '"next month" -> February 2023' in month_hint
    assert AnswerGenerator._normalise_temporal_answer(
        "next month", month_hint
    ) == "February 2023"
    week_hint = AnswerGenerator._calendar_hints(
        "I had the picnic last week.",
        "1:00 pm on 6 July, 2023",
    )
    assert AnswerGenerator._normalise_temporal_answer(
        "29 June 2023", week_hint
    ) == "the week before 6 July 2023"


def test_temporal_canonicalization_uses_query_anchored_evidence():
    from memrewire.core.query_context import QueryContext, ContextEvidence

    query_context = QueryContext(
        query_context_id="ws-test",
        query="When did Caroline have a picnic?",
        question_type="temporal",
        evidence=[
            ContextEvidence(
                evidence_id="W1", cell_id="t1", node_id="n1",
                text="Gina designed hoodies last week.",
                timestamp="21 June 2023",
            ),
            ContextEvidence(
                evidence_id="W2", cell_id="t2", node_id="n2",
                text="Caroline had a picnic last week.",
                timestamp="6 July 2023",
            ),
        ],
    )
    assert AnswerGenerator._normalise_query_context_temporal_answer(
        "29 June 2023",
        query_context.query,
        query_context,
    ) == "the week before 6 July 2023"


if __name__ == "__main__":
    test_cell_builder_preserves_missing_turn_provenance()
    test_event_bundle_creates_multiple_contiguous_cells()
    test_local_query_context_selects_connected_multi_evidence()
    test_same_node_role_changes_with_query()
    test_role_binding_is_task_time_and_decomposes_gain()
    test_same_memory_node_binds_as_anchor_or_bridge_by_task()
    test_retriever_routes_once_and_assembles_locally()
    test_default_readout_is_field_derived_without_a_cell_guard()
    test_query_similarity_readout_skips_rollout_and_uses_direct_top_k()
    test_shared_local_transition_is_permutation_equivariant()
    test_prompt_serialization_drops_partial_relations()
    test_transient_assembly_is_query_local_and_persistent_store_is_unchanged()
    test_locomo_calendar_hint_uses_session_anchor_without_iso_format()
    test_graph_coverage_requires_content_observability()
    test_structured_answer_is_normalized_to_plain_text()
    print("cell query_context tests passed")
