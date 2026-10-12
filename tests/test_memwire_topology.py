"""Minimal invariants for MemWire's write-time topology compiler.

These tests deliberately exercise only the dependency-light graph layer.  The
compiler must change the fixed wiring objective without changing the storage
budget or introducing query-time state into the persistent graph.
"""

import inspect

from memwire.core.models.memory_entry import InteractionThread, ThreadNode
from memwire.core.thread_store import ThreadStore


def _threads():
    vectors = [
        [1.00, 0.00, 0.10, 0.00],
        [0.92, 0.18, 0.10, 0.00],
        [0.80, 0.32, 0.10, 0.00],
        [0.00, 1.00, 0.10, 0.00],
        [0.18, 0.92, 0.10, 0.00],
        [0.32, 0.80, 0.10, 0.00],
    ]
    return [
        InteractionThread(
            thread_id=f"t{index}",
            head=f"topic {index}",
            session_id="s0" if index < 3 else "s1",
            ordinal=index,
            nodes=[ThreadNode(node_id=f"t{index}:n0", text=f"turn {index}")],
            embedding=vector,
        )
        for index, vector in enumerate(vectors)
    ]


def _pairs(threads):
    pairs = set()
    for thread in threads:
        for link in thread.links:
            left, right = thread.thread_id, link.target_thread_id
            pairs.add(tuple(sorted((left, right))))
    return pairs


def _chronology(threads):
    ordered = sorted(threads, key=lambda item: (item.ordinal, item.thread_id))
    pairs = set()
    for left, right in zip(ordered, ordered[1:]):
        pairs.add(tuple(sorted((left.thread_id, right.thread_id))))
    return pairs


def _optional_degrees(threads):
    chronology = _chronology(threads)
    degrees = {thread.thread_id: 0 for thread in threads}
    for left, right in _pairs(threads) - chronology:
        degrees[left] += 1
        degrees[right] += 1
    return degrees


def _link_signature(threads):
    return sorted(
        (thread.thread_id, link.target_thread_id, link.relation, round(link.weight, 12))
        for thread in threads
        for link in thread.links
    )


def _clone_threads():
    return [
        thread.model_copy(deep=True)
        if hasattr(thread, "model_copy")
        else thread.copy(deep=True)
        for thread in _threads()
    ]


def _compile(threads):
    return ThreadStore._compile_operator_links(
        threads,
        max_neighbors=2,
        min_similarity=0.0,
        candidate_multiplier=3,
        sketch_dim=6,
        swap_search_width=64,
        max_swaps=4,
    )


def test_operator_compiler_preserves_legacy_edge_budget_and_optional_degrees():
    legacy = _clone_threads()
    ThreadStore._build_thread_links(
        legacy,
        max_neighbors=2,
        min_similarity=0.0,
    )
    baseline_pairs = _pairs(legacy)
    baseline_degrees = _optional_degrees(legacy)

    compiled = _clone_threads()
    diagnostics = _compile(compiled)

    assert len(_pairs(compiled)) == len(baseline_pairs)
    assert diagnostics["baseline_edge_budget"] == len(baseline_pairs)
    assert diagnostics["selected_edges"] == len(baseline_pairs)
    assert diagnostics["degree_cap_satisfied"]
    assert _optional_degrees(compiled) == baseline_degrees


def test_operator_compiler_is_deterministic_for_identical_input():
    first = _clone_threads()
    second = _clone_threads()

    first_diagnostics = _compile(first)
    second_diagnostics = _compile(second)

    assert first_diagnostics == second_diagnostics
    assert _link_signature(first) == _link_signature(second)


def test_operator_compiler_has_no_query_or_answer_inputs():
    parameters = set(inspect.signature(ThreadStore._compile_operator_links).parameters)
    assert parameters.isdisjoint({"query", "question", "answer", "evidence"})
    assert parameters == {
        "threads",
        "max_neighbors",
        "min_similarity",
        "candidate_multiplier",
        "sketch_dim",
        "objective_scale",
        "swap_search_width",
        "max_swaps",
    }


def test_random_rewiring_preserves_edge_budget_chronology_and_degrees():
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    threads = [
        InteractionThread(
            thread_id=f"r{index}",
            head=f"topic {index}",
            session_id="s0",
            ordinal=index,
            nodes=[ThreadNode(node_id=f"r{index}:n0", text=f"turn {index}")],
            embedding=vector,
        )
        for index, vector in enumerate(vectors)
    ]
    legacy = [
        thread.model_copy(deep=True)
        if hasattr(thread, "model_copy")
        else thread.copy(deep=True)
        for thread in threads
    ]
    ThreadStore._build_thread_links(
        legacy,
        max_neighbors=1,
        min_similarity=0.99,
    )
    baseline_degrees = _optional_degrees(legacy)
    baseline_edge_count = len(_pairs(legacy))

    diagnostics = ThreadStore._random_rewire_links(
        threads,
        max_neighbors=1,
        min_similarity=0.99,
        seed=17,
        swaps_per_edge=2,
    )

    assert diagnostics["accepted_swaps"] > 0
    assert diagnostics["degree_sequence_preserved"]
    assert diagnostics["selected_edges"] == baseline_edge_count
    assert len(_pairs(threads)) == baseline_edge_count
    assert _optional_degrees(threads) == baseline_degrees


def test_random_rewiring_is_reproducible_for_a_fixed_seed():
    def build():
        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        return [
            InteractionThread(
                thread_id=f"r{index}",
                head=f"topic {index}",
                nodes=[ThreadNode(node_id=f"r{index}:n0", text=f"turn {index}")],
                embedding=vector,
            )
            for index, vector in enumerate(vectors)
        ]

    first, second = build(), build()
    first_diagnostics = ThreadStore._random_rewire_links(
        first, max_neighbors=1, min_similarity=0.99, seed=23, swaps_per_edge=2
    )
    second_diagnostics = ThreadStore._random_rewire_links(
        second, max_neighbors=1, min_similarity=0.99, seed=23, swaps_per_edge=2
    )

    assert first_diagnostics == second_diagnostics
    assert _link_signature(first) == _link_signature(second)
