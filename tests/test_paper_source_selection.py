"""Checks for the source-turn selection contract used by the paper method."""

from memwire.core.workspace import TransientAssemblyBuilder
from memwire.core.answer_generator import AnswerGenerator
from memwire.core.workspace import MemoryWorkspace, WorkspaceEvidence
from memwire.core.settings import settings as config


def _candidate(node_id, score, source_ordinal):
    return {
        "thread_id": "cell",
        "node_id": node_id,
        "text": node_id,
        "thread_head": "head",
        "source_turn_ids": [node_id],
        "ordinal": source_ordinal,
        "thread_ordinal": 0,
        "source_ordinal": source_ordinal,
        "timestamp": None,
        "dense_score": score,
        "query_affinity": score,
        "route_score": 1.0,
        "thread_rank": 1,
        "activation_score": score,
        "node_embedding": [score, 1.0],
    }


def test_query_similarity_selects_at_most_k_source_nodes_then_orders_by_source_position():
    builder = TransientAssemblyBuilder(max_readout_cells=2, readout_mode="query_similarity")
    selected, _, _ = builder._readout(
        [_candidate("turn-3", 0.99, 3), _candidate("turn-1", 0.90, 1), _candidate("turn-2", 0.80, 2)],
        "general",
    )
    assert [item["node_id"] for item in selected] == ["turn-1", "turn-3"]


def test_source_only_context_keeps_cell_source_and_time_metadata(monkeypatch):
    monkeypatch.setattr(config, "WORKSPACE_SOURCE_ONLY_SERIALIZATION", True, raising=False)
    workspace = MemoryWorkspace(
        workspace_id="w",
        query="q",
        question_type="general",
        evidence=[
            WorkspaceEvidence(
                evidence_id="W1",
                thread_id="cell-1",
                node_id="n1",
                text="Alice visited Rome.",
                source_turn_ids=["D1:3"],
                timestamp="2024-01-02",
            )
        ],
    )
    rendered = AnswerGenerator._format_workspace(workspace)
    assert "cell=cell-1" in rendered
    assert "source=D1:3" in rendered
    assert "time=2024-01-02" in rendered
