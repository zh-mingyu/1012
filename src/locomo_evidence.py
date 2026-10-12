"""Normalize LoCoMo evidence annotations to source-turn IDs."""

from __future__ import annotations

import re
from typing import Any


_SEPARATORS = re.compile(r"[;,\s]+")
_MISSING_SESSION_DIGIT = re.compile(r"^D:(\d+:\d+)$")
_TURN_ID = re.compile(r"^D(\d+):(\d+)$")


def parse_evidence_ids(evidence: Any) -> tuple[list[str], dict[str, str]]:
    values = evidence if isinstance(evidence, (list, tuple, set)) else [evidence]
    ids = set()
    corrections = {}
    for value in values:
        if value is None:
            continue
        for token in _SEPARATORS.split(str(value).strip()):
            if not token:
                continue
            missing_digit = _MISSING_SESSION_DIGIT.fullmatch(token)
            normalized = f"D{missing_digit.group(1)}" if missing_digit else token
            ids.add(normalized)
            if normalized != token:
                corrections[token] = normalized
    return sorted(ids), corrections


def normalize_evidence_ids(evidence: Any) -> list[str]:
    return parse_evidence_ids(evidence)[0]


def resolve_source_turn_ids(
    evidence_ids: list[str], source_turn_ids: set[str]
) -> tuple[list[str], list[str], dict[str, str]]:
    """Resolve exact IDs and unambiguous numeric-padding variants."""
    numeric_sources: dict[tuple[int, int], list[str]] = {}
    for source_id in source_turn_ids:
        match = _TURN_ID.fullmatch(source_id)
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            numeric_sources.setdefault(key, []).append(source_id)

    resolved = set()
    unresolved = set()
    corrections = {}
    for evidence_id in evidence_ids:
        if evidence_id in source_turn_ids:
            resolved.add(evidence_id)
            continue
        match = _TURN_ID.fullmatch(evidence_id)
        candidates = (
            numeric_sources.get((int(match.group(1)), int(match.group(2))), [])
            if match else []
        )
        if len(candidates) == 1:
            resolved.add(candidates[0])
            corrections[evidence_id] = candidates[0]
        else:
            unresolved.add(evidence_id)
    return sorted(resolved), sorted(unresolved), corrections
