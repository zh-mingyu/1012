"""Answer synthesis from a transient query-local memory assembly."""

from __future__ import annotations

from datetime import datetime, timedelta
import math
import re
from typing import TYPE_CHECKING, List, Union

from memwire.core.models.memory_entry import MemoryEntry
from memwire.core.settings import settings as config
if TYPE_CHECKING:
    from memwire.core.utils.llm_client import LLMClient
from memwire.core.workspace import MemoryWorkspace


_MAX_ANSWER_ATTEMPTS = 1


class AnswerGenerator:
    """Render assembled source cells with provenance and ask once."""

    _QUERY_STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "before", "did", "does", "for",
        "from", "had", "has", "have", "her", "his", "how", "in", "is", "many",
        "of", "on", "the", "their", "to", "was", "were", "what", "when", "which",
        "with",
    }
    _NUMBER_WORDS = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
        5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
        10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
        14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
        18: "eighteen", 19: "nineteen", 20: "twenty",
    }

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    @staticmethod
    def _answer_text(value: object, fallback: str) -> str:
        """Normalize structured model answers to the benchmark text contract."""
        if isinstance(value, list):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        if isinstance(value, dict):
            return ", ".join(
                str(item).strip() for item in value.values() if str(item).strip()
            )
        if value is None:
            return fallback.strip()
        return str(value).strip()

    @classmethod
    def _content_tokens(cls, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", str(text).lower())) - cls._QUERY_STOPWORDS

    @classmethod
    def _normalise_count_answer(cls, answer: str, query: str) -> str:
        """Use a stable lexical count form for token-based answer contracts."""
        value = str(answer or "").strip()
        if not str(query or "").lower().startswith("how many"):
            return value
        match = re.fullmatch(r"(\d+)\.?", value)
        if not match:
            return value
        number = int(match.group(1))
        return cls._NUMBER_WORDS.get(number, value)

    @classmethod
    def _normalise_workspace_temporal_answer(
        cls,
        answer: str,
        query: str,
        workspace: MemoryWorkspace,
    ) -> str:
        """Use the most query-anchored temporal evidence for canonicalization."""
        query_tokens = cls._content_tokens(query)
        candidates = []
        for item in workspace.evidence:
            hint = cls._calendar_hints(item.text, item.timestamp)
            rewritten = cls._normalise_temporal_answer(answer, hint)
            if rewritten == answer:
                continue
            source_tokens = cls._content_tokens(item.text)
            overlap = len(query_tokens & source_tokens)
            union = len(query_tokens | source_tokens) or 1
            candidates.append((overlap, overlap / union, rewritten, item.node_id))
        if not candidates:
            return answer
        overlap, _jaccard, rewritten, _node_id = max(candidates)
        return rewritten if overlap >= 1 else answer

    @staticmethod
    def _normalise_temporal_answer(answer: str, context: str) -> str:
        """Canonicalize dates to the temporal granularity exposed by evidence.

        Relative phrases denote equivalence classes of calendar instants (a
        week, weekend, or weekday relation), rather than an arbitrary single
        day.  When the model emits the resolved anchor, map it back to the
        evidence phrase so answer generation preserves that relation.
        """
        value = str(answer or "").strip()
        lowered = value.lower()
        if not value:
            return value
        replacements = [
            (r'"last year" -> (\d{4})', lambda m: m.group(1)),
            (r'"next year" -> (\d{4})', lambda m: m.group(1)),
            (r'"last month" -> ([A-Za-z]+\s+\d{4})', lambda m: m.group(1)),
            (r'"next month" -> ([A-Za-z]+\s+\d{4})', lambda m: m.group(1)),
            (r'"yesterday" -> ([^;]+)', lambda m: m.group(1).strip()),
            (r'"tomorrow" -> ([^;]+)', lambda m: m.group(1).strip()),
        ]
        for pattern, resolver in replacements:
            match = re.search(pattern, context, flags=re.IGNORECASE)
            if not match:
                continue
            trigger = pattern.split('"')[1].lower()
            if lowered == trigger or lowered.rstrip(".") == trigger:
                return resolver(match)
        for match in re.finditer(
            r'"([^"]+)" -> preserve relation=([^;\n]+); '
            r'(?:resolved date|approximate anchor)=([^;\n]+)',
            context,
            flags=re.IGNORECASE,
        ):
            trigger = match.group(1).strip().lower()
            relation = match.group(2).strip()
            resolved = AnswerGenerator._parse_timestamp(match.group(3).strip())
            if lowered.rstrip(".") == trigger.rstrip("."):
                return relation
            answer_date = AnswerGenerator._parse_timestamp(value)
            if (
                answer_date is not None
                and resolved is not None
                and answer_date.date() == resolved.date()
            ):
                return relation
        answer_date = AnswerGenerator._parse_timestamp(value)
        if answer_date is not None:
            for match in re.finditer(
                r'"([^"]+)" -> preserve phrase; '
                r'(?:resolved date|approximate anchor)=([^;\n]+)',
                context,
                flags=re.IGNORECASE,
            ):
                resolved = AnswerGenerator._parse_timestamp(match.group(2).strip())
                if resolved is not None and answer_date.date() == resolved.date():
                    return match.group(1).strip()
        return value

    @staticmethod
    def _canonicalize_date_spacing(answer: str) -> str:
        """Remove formatting-only zero padding from natural-language dates."""
        value = str(answer or "")
        return re.sub(
            r"\b0([1-9])(?=\s+[A-Za-z]+\s+\d{4}\b)",
            r"\1",
            value,
        )

    def generate_answer(
        self,
        query: str,
        contexts: Union[List[MemoryEntry], MemoryWorkspace],
    ) -> str:
        if not contexts:
            return "No relevant information found"
        context = (
            self._format_workspace(contexts, query=query)
            if isinstance(contexts, MemoryWorkspace)
            else self._format_flat(contexts)
        )
        question_type = (
            contexts.question_type
            if isinstance(contexts, MemoryWorkspace)
            else "general"
        )
        prompt = self._build_prompt(query, context, question_type=question_type)
        messages = [
            {"role": "system", "content": "Answer strictly from the supplied source cells and return valid JSON."},
            {"role": "user", "content": prompt},
        ]
        input_limit = getattr(config, "ANSWER_MAX_INPUT_TOKENS", None)
        budget_dropped_cells = 0
        counted_prompt_tokens = None
        if input_limit:
            if not isinstance(contexts, MemoryWorkspace):
                raise RuntimeError(
                    "ANSWER_MAX_INPUT_TOKENS requires a token-countable MemoryWorkspace"
                )
            template_kwargs = {}
            if "qwen" in str(self.llm_client.model or "").lower():
                template_kwargs["enable_thinking"] = bool(
                    self.llm_client.enable_thinking
                    and not getattr(config, "USE_JSON_FORMAT", False)
                )
            counted_prompt_tokens = self.llm_client.count_prompt_tokens(
                messages,
                template_kwargs=template_kwargs,
            )
            if counted_prompt_tokens is None:
                raise RuntimeError(
                    "ANSWER_MAX_INPUT_TOKENS is set but no local tokenizer is configured"
                )
            minimum_cells = max(1, int(getattr(config, "WORKSPACE_MIN_VISIBLE_CELLS", 1)))
            while (
                counted_prompt_tokens > int(input_limit)
                and len(contexts.evidence) > minimum_cells
            ):
                drop_index = min(
                    range(len(contexts.evidence)),
                    key=lambda index: (
                        float(getattr(contexts.evidence[index], "readout_score", 0.0)),
                        -index,
                    ),
                )
                contexts.evidence.pop(drop_index)
                budget_dropped_cells += 1
                context = self._format_workspace(contexts, query=query)
                prompt = self._build_prompt(query, context, question_type=question_type)
                messages[1]["content"] = prompt
                counted_prompt_tokens = self.llm_client.count_prompt_tokens(
                    messages,
                    template_kwargs=template_kwargs,
                )
            if counted_prompt_tokens > int(input_limit):
                raise RuntimeError(
                    f"Minimum-evidence answer prompt is {counted_prompt_tokens} tokens, "
                    f"above ANSWER_MAX_INPUT_TOKENS={int(input_limit)}"
                )
        response = ""
        last_error: Exception | None = None
        for attempt in range(_MAX_ANSWER_ATTEMPTS):
            try:
                response_format = {"type": "json_object"} if getattr(config, "USE_JSON_FORMAT", False) else None
                with self.llm_client.usage_scope(
                    prompt_budget_dropped_cells=budget_dropped_cells,
                    tokenizer_prompt_tokens=counted_prompt_tokens,
                ):
                    response = self.llm_client.chat_completion(
                        messages,
                        temperature=0.1,
                        response_format=response_format,
                        max_retries=1,
                        stage="answer",
                    )
            except Exception as error:
                last_error = error
                if attempt + 1 == _MAX_ANSWER_ATTEMPTS:
                    return f"Failed to generate answer: {error}"
                continue

            try:
                result = self.llm_client.extract_json(response)
            except Exception as error:
                last_error = error
                if attempt + 1 == _MAX_ANSWER_ATTEMPTS:
                    return response.strip() or f"Failed to generate answer: {error}"
                continue

            try:
                answer = self._answer_text(result.get("answer"), response)
                if (
                    getattr(config, "ENABLE_TEMPORAL_CALIBRATION", True)
                    and "temporal" in str(question_type).lower()
                ):
                    if isinstance(contexts, MemoryWorkspace):
                        answer = self._normalise_workspace_temporal_answer(
                            answer,
                            query,
                            contexts,
                        )
                    else:
                        answer = self._normalise_temporal_answer(answer, context)
                    answer = self._canonicalize_date_spacing(answer)
                if getattr(config, "ENABLE_COUNT_ANSWER_NORMALIZATION", True):
                    answer = self._normalise_count_answer(answer, query)
                return answer
            except Exception as error:
                last_error = error
                break
        return response.strip() or f"Failed to generate answer: {last_error}"

    @staticmethod
    def _format_flat(contexts: List[MemoryEntry]) -> str:
        return "\n\n".join(
            f"[Evidence {index}] {entry.lossless_restatement}"
            + (f" (time: {entry.timestamp})" if entry.timestamp else "")
            + (f" [source: {', '.join(entry.source_turn_ids)}]" if entry.source_turn_ids else "")
            for index, entry in enumerate(contexts, 1)
        )

    @staticmethod
    def _parse_timestamp(timestamp: str | None) -> datetime | None:
        """Parse ISO timestamps and LoCoMo's natural-language session dates."""
        if not timestamp:
            return None
        value = str(timestamp).strip()
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            pass
        date_text = re.sub(r"^.*?\bon\s+", "", value, flags=re.IGNORECASE)
        date_text = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", date_text, flags=re.IGNORECASE)
        for pattern in ("%d %B, %Y", "%d %B %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(date_text, pattern)
            except ValueError:
                continue
        return None

    @classmethod
    def _calendar_hints(cls, text: str, timestamp: str | None) -> str:
        """Add deterministic anchors for relative dates in original turns."""
        if not text or not timestamp:
            return ""
        anchor = cls._parse_timestamp(timestamp)
        if anchor is None:
            return ""
        hints = []
        lowered = text.lower()
        if "last year" in lowered:
            hints.append(f'"last year" -> {anchor.year - 1}')
        if "next year" in lowered:
            hints.append(f'"next year" -> {anchor.year + 1}')
        if "yesterday" in lowered:
            hints.append(f'"yesterday" -> {(anchor - timedelta(days=1)).strftime("%d %B %Y")}')
        if "tomorrow" in lowered:
            hints.append(f'"tomorrow" -> {(anchor + timedelta(days=1)).strftime("%d %B %Y")}')
        anchor_label = f"{anchor.day} {anchor.strftime('%B %Y')}"
        if "last week" in lowered:
            hints.append(
                f'"last week" -> preserve relation=the week before {anchor_label}; '
                f'approximate anchor={(anchor - timedelta(days=7)).strftime("%d %B %Y")}'
            )
        if "next week" in lowered:
            hints.append(
                f'"next week" -> preserve relation=the week after {anchor_label}; '
                f'approximate anchor={(anchor + timedelta(days=7)).strftime("%d %B %Y")}'
            )
        if "last weekend" in lowered or "this past weekend" in lowered:
            trigger = "last weekend" if "last weekend" in lowered else "this past weekend"
            hints.append(
                f'"{trigger}" -> preserve relation=the weekend before {anchor_label}; '
                f'approximate anchor={(anchor - timedelta(days=7)).strftime("%d %B %Y")}'
            )
        if "two weekends ago" in lowered:
            hints.append(
                f'"two weekends ago" -> preserve relation=two weekends before {anchor_label}; '
                f'approximate anchor={(anchor - timedelta(days=14)).strftime("%d %B %Y")}'
            )
        if "last month" in lowered:
            year = anchor.year - (1 if anchor.month == 1 else 0)
            month = 12 if anchor.month == 1 else anchor.month - 1
            hints.append(f'"last month" -> {datetime(year, month, 1).strftime("%B %Y")}')
        if "next month" in lowered:
            year = anchor.year + (1 if anchor.month == 12 else 0)
            month = 1 if anchor.month == 12 else anchor.month + 1
            hints.append(f'"next month" -> {datetime(year, month, 1).strftime("%B %Y")}')
        weekday_index = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        for weekday, target in weekday_index.items():
            if f"last {weekday}" in lowered:
                delta = (anchor.weekday() - target) % 7 or 7
                resolved = anchor - timedelta(days=delta)
                hints.append(
                    f'"last {weekday}" -> preserve relation=the {weekday.title()} before '
                    f'{anchor_label}; resolved date={resolved.strftime("%d %B %Y")}'
                )
        weekday_abbreviations = {
            "mon": "monday", "tue": "tuesday", "tues": "tuesday",
            "wed": "wednesday", "thu": "thursday", "thur": "thursday",
            "thurs": "thursday", "fri": "friday", "sat": "saturday",
            "sun": "sunday",
        }
        for abbreviation, weekday in weekday_abbreviations.items():
            if re.search(rf"\blast {abbreviation}\.?\b", lowered):
                target = weekday_index[weekday]
                delta = (anchor.weekday() - target) % 7 or 7
                resolved = anchor - timedelta(days=delta)
                hints.append(
                    f'"last {abbreviation}" -> preserve relation=the {weekday.title()} before '
                    f'{anchor_label}; resolved date={resolved.strftime("%d %B %Y")}'
                )
        # Preserve explicit relative phrases while exposing a deterministic
        # calendar anchor.  The phrase itself is often the benchmark's
        # expected granularity (e.g. "the Friday before 15 July 2023").
        inline_date = r"(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})"
        for match in re.finditer(
            rf"\bthe\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+before\s+{inline_date}",
            text,
            flags=re.IGNORECASE,
        ):
            target_date = cls._parse_timestamp(match.group(2))
            if target_date:
                weekday = match.group(1).title()
                delta = (target_date.weekday() - weekday_index[weekday.lower()]) % 7 or 7
                resolved = target_date - timedelta(days=delta)
                phrase = match.group(0)
                hints.append(
                    f'"{phrase}" -> preserve phrase; resolved date={resolved.strftime("%d %B %Y")}'
                )
        for match in re.finditer(
            rf"\b(two\s+weekends?|the\s+weekend|the\s+week)\s+before\s+{inline_date}",
            text,
            flags=re.IGNORECASE,
        ):
            target_date = cls._parse_timestamp(match.group(2))
            if target_date:
                phrase = match.group(0)
                offset = 14 if match.group(1).lower().startswith("two") else 7
                resolved = target_date - timedelta(days=offset)
                hints.append(
                    f'"{phrase}" -> preserve phrase; approximate anchor={resolved.strftime("%d %B %Y")}'
                )
        for match in re.finditer(
            rf"\bthe\s+week\s+of\s+{inline_date}",
            text,
            flags=re.IGNORECASE,
        ):
            hints.append(f'"{match.group(0)}" -> preserve phrase')
        return "; calendar-hint=" + "; ".join(hints) if hints else ""

    @classmethod
    def _format_workspace(cls, workspace: MemoryWorkspace, query: str = "") -> str:
        compact = bool(getattr(config, "WORKSPACE_COMPACT_SERIALIZATION", True))
        source_only = bool(getattr(config, "WORKSPACE_SOURCE_ONLY_SERIALIZATION", False))
        if source_only:
            temporal_query = "temporal" in str(workspace.question_type or "").lower()
            char_budget = max(0, int(getattr(config, "WORKSPACE_SOURCE_CHAR_BUDGET", 900)))
            lines: List[str] = []
            used_chars = 0
            serialized_ids: List[str] = []
            # Selection is scored before presentation is reordered by source
            # time. If serialization must trim text, retain the highest-ranked
            # source turns first, then restore the original temporal order.
            ranked_items = sorted(
                workspace.assembly_cells,
                key=lambda item: (
                    -float(getattr(item, "readout_score", 0.0)),
                    str(getattr(item, "evidence_id", "")),
                ),
            )
            retained_ids: set[str] = set()
            for item in ranked_items:
                text = re.sub(r"\s+", " ", str(item.text or "")).strip()
                if not text:
                    continue
                source = ", ".join(item.source_turn_ids) or "unknown"
                metadata = f"[cell={item.thread_id}; source={source}"
                if item.timestamp:
                    metadata += f"; time={item.timestamp}"
                metadata += "]"
                suffix = ""
                if temporal_query and item.timestamp:
                    hints = cls._calendar_hints(item.text, item.timestamp)
                    if hints:
                        suffix = f" {hints[:180]}"
                block = f"{metadata} {text}{suffix}"
                if char_budget and retained_ids and used_chars + len(block) + 1 > char_budget:
                    continue
                retained_ids.add(str(item.evidence_id))
                used_chars += len(block) + 1
            for item in workspace.assembly_cells:
                if str(item.evidence_id) not in retained_ids:
                    continue
                text = re.sub(r"\s+", " ", str(item.text or "")).strip()
                source = ", ".join(item.source_turn_ids) or "unknown"
                metadata = f"[cell={item.thread_id}; source={source}"
                if item.timestamp:
                    metadata += f"; time={item.timestamp}"
                metadata += "]"
                suffix = ""
                if temporal_query and item.timestamp:
                    hints = cls._calendar_hints(item.text, item.timestamp)
                    if hints:
                        suffix = f" {hints[:180]}"
                lines.append(f"{metadata} {text}{suffix}")
                serialized_ids.append(item.evidence_id)
            if not lines:
                lines.append("No visible source cell.")
            workspace.serialized_cell_count = len(serialized_ids)
            return "\n".join(lines)
        if compact:
            lines = [
                "Source cells from the transient workspace (roles are temporary):",
            ]
            # The default address is the question itself. Repeating it costs
            # tokens without adding evidence; retain it only for an explicit
            # planner-produced address.
            if workspace.address and workspace.address.strip() != str(query or "").strip():
                lines.append(f"Address: {workspace.address}")
        else:
            lines = []
        if not compact:
            # Keep the pre-compact answer contract available as a controlled
            # quality reference. The recurrent field is unchanged; only its
            # textual interface differs between the two modes.
            lines = [
                "Transient role-bound state (query-local; source substrate unchanged):",
                f"Address: {workspace.address}",
                f"Question type: {workspace.question_type}; complexity: {workspace.complexity}",
            ]
            lines.append(
                f"Stable activation readout (visibility cutoff={workspace.visibility_cutoff:.3f}; "
                f"field coverage diagnostic={workspace.assembly_coverage:.3f}; "
                f"steps={workspace.rollout_steps}; delta={workspace.state_delta:.4f}; "
                f"mode={workspace.dynamics_mode}):"
            )
            guard = int(
                getattr(workspace, "prompt_guard_tokens", 0)
                or getattr(config, "WORKSPACE_PROMPT_GUARD_TOKENS", 0)
            )
            serialized_ids: List[str] = []
            estimated_tokens = sum(max(1, len(line) // 4) for line in lines)
            for item in workspace.assembly_cells:
                source = ", ".join(item.source_turn_ids) or "unknown"
                source_time = f"; source-time={item.timestamp}" if item.timestamp else ""
                hints = (
                    cls._calendar_hints(item.text, item.timestamp)
                    if getattr(config, "ENABLE_TEMPORAL_CALIBRATION", True)
                    else ""
                )
                block = (
                    f"[Assembly cell {item.evidence_id}] state={item.activation_score:.3f}; "
                    f"seed={item.seed_affinity:.3f}; support={item.support_influence:.3f}; "
                    f"inhibition={item.inhibition_score:.3f}; thread={item.thread_id}; "
                    f"role-vector=(drive={item.query_drive:.3f}, "
                    f"local-support={item.local_support:.3f}, "
                    f"relay={item.relay_mediation:.3f}); "
                    f"source={source}{source_time}{hints}\n"
                    f"Trajectory: {item.thread_head}\n{item.text}"
                )
                block_tokens = max(1, len(block) // 4)
                if guard and serialized_ids and estimated_tokens + block_tokens > guard:
                    continue
                lines.append(block)
                serialized_ids.append(item.evidence_id)
                estimated_tokens += block_tokens
            if len(serialized_ids) < len(workspace.assembly_cells):
                omitted = len(workspace.assembly_cells) - len(serialized_ids)
                lines.append(
                    f"[Serialization note] {omitted} lower-activation cells omitted from this prompt."
                )
            workspace.serialized_cell_count = len(serialized_ids)
            if workspace.relations:
                lines.append("Local relations:")
                for relation in workspace.relations:
                    if (
                        not serialized_ids
                        or relation.source_evidence_id not in serialized_ids
                        or relation.target_evidence_id not in serialized_ids
                    ):
                        continue
                    lines.append(
                        f"- {relation.source_evidence_id} -> {relation.target_evidence_id}: "
                        f"{relation.relation_type}"
                    )
            lines.append(
                "Role-vector values describe query-local support and mediation; use "
                "them only to organize reasoning and verify every claim in the source cells."
            )
            return "\n".join(lines)

        # The compact branch above is the experimental lower-token interface.
        # It is deliberately kept separate from the high-fidelity reference.
        guard = int(
            getattr(workspace, "prompt_guard_tokens", 0)
            or getattr(config, "WORKSPACE_PROMPT_GUARD_TOKENS", 0)
        )
        temporal_query = "temporal" in str(workspace.question_type or "").lower()
        serialized_ids: List[str] = []
        seen_threads: set[str] = set()
        estimated_tokens = sum(max(1, int(math.ceil(len(line) / 3.0))) for line in lines)
        for item in workspace.assembly_cells:
            source = ", ".join(item.source_turn_ids) or "unknown"
            source_time = f"|time={item.timestamp}" if temporal_query and item.timestamp else ""
            hints = (
                cls._calendar_hints(item.text, item.timestamp)
                if temporal_query and getattr(config, "ENABLE_TEMPORAL_CALIBRATION", True)
                else ""
            )
            thread_id = str(item.thread_id)
            thread_header = ""
            if thread_id not in seen_threads:
                head_limit = 72
                head = re.sub(r"\s+", " ", str(item.thread_head or "")).strip()[:head_limit]
                thread_header = f"T{thread_id}: {head}\n" if head else f"T{thread_id}:\n"
            role = str(item.functional_role or item.role or "context")
            block = f"{thread_header}[Assembly cell {item.evidence_id}] role={role}{source_time}{hints} {item.text}"
            block_tokens = max(1, int(math.ceil(len(block) / 3.0)))
            if guard and serialized_ids and estimated_tokens + block_tokens > guard:
                continue
            if thread_header:
                lines.append(thread_header.rstrip("\n"))
                seen_threads.add(thread_id)
            lines.append(block[len(thread_header):] if thread_header else block)
            serialized_ids.append(item.evidence_id)
            estimated_tokens += block_tokens
        if len(serialized_ids) < len(workspace.assembly_cells):
            omitted = len(workspace.assembly_cells) - len(serialized_ids)
            lines.append(f"[Omitted {omitted} lower-utility cells due to the workspace token budget]")
        workspace.serialized_cell_count = len(serialized_ids)
        if workspace.relations:
            lines.append("Local relations:")
            for relation in workspace.relations:
                if (
                    not serialized_ids
                    or relation.source_evidence_id not in serialized_ids
                    or relation.target_evidence_id not in serialized_ids
                ):
                    continue
                lines.append(
                    f"{relation.source_evidence_id} -> {relation.target_evidence_id} "
                    f"({relation.relation_type})"
                )
        return "\n".join(lines)

    @staticmethod
    def _format_contexts(
        contexts: Union[List[MemoryEntry], MemoryWorkspace],
        query: str = "",
    ) -> str:
        """Compatibility adapter used by the LoCoMo evaluator."""
        return (
            AnswerGenerator._format_workspace(contexts, query=query)
            if isinstance(contexts, MemoryWorkspace)
            else AnswerGenerator._format_flat(contexts)
        )

    @staticmethod
    def _build_prompt(query: str, context: str, question_type: str = "general") -> str:
        mode = str(question_type or "general").lower()
        temporal_calibration = getattr(config, "ENABLE_TEMPORAL_CALIBRATION", True)
        prompt_style = str(getattr(config, "ANSWER_PROMPT_STYLE", "standard")).lower()
        if prompt_style in {"minimal", "source_only"}:
            if "temporal" in mode:
                instruction = "For dates, use the event wording; do not treat a session date as the event date."
            elif "set-valued" in mode or "multi" in mode:
                instruction = "Collect all distinct supported items or hops; do not invent missing items."
            else:
                instruction = "Use only facts entailed by the evidence; if none, answer Not mentioned in the conversation."
            return (
                f"Q: {query}\n"
                f"Evidence:\n{context}\n"
                f"{instruction}\n"
                'Return JSON only: {"answer":"concise answer"}'
            )
        if "temporal" in mode and temporal_calibration:
            mode_instruction = """
This is a temporal question. A source-time is only the session anchor, not the
event time. Resolve the event time from the turn wording and calendar-hint.
If the hint resolves "last year", "yesterday", or "tomorrow", output the
resolved year/date rather than the literal relative word. If the evidence
explicitly says "the Friday/week/weekend before <date>", preserve that phrase
in the answer because it is the entailed granularity; do not substitute the
session date. Never treat source-time as the event time.
"""
        elif "set-valued" in mode or "multi" in mode:
            mode_instruction = """
This question requires source-cell composition. Inspect every visible trajectory,
collect all distinct supported items or hops, and return a complete deduplicated
list. Do not stop after the first relevant evidence item, and do not fill a
missing item with a plausible but unsupported guess. The JSON answer value must
be one concise string, not an array.
"""
        else:
            mode_instruction = ""
        temporal_instruction = """
Each source cell may include a source-time (the session date) and a
calendar-hint. Use that date and hint to resolve relative expressions in the
turn text (for example, yesterday, last year, or last Saturday) and report the
resulting date or year when the question asks when. Preserve the temporal
granularity supported by the evidence.
""" if temporal_calibration and "temporal" in mode else ""
        if getattr(config, "WORKSPACE_COMPACT_SERIALIZATION", True):
            return f"""
Answer the question using only the transient role-bound state below.

Question: {query}

{context}

{mode_instruction}
{temporal_instruction}

Give a concise answer, preferably a short entity, value, or date rather than a
paragraph. Resolve multi-step questions by following the displayed local
relations, and do not add facts absent from the source text. A cell counts as
support only when it entails the requested fact; unrelated visible cells are
not support. If no visible cell entails the target, answer exactly "Not
mentioned in the conversation". Return ONLY JSON:
{{"reasoning": "brief source-grounded explanation", "answer": "concise answer"}}
"""
        return f"""
        Answer the question using only the transient role-bound state below.

Question: {query}

{context}

{mode_instruction}

Give a concise answer, preferably a short entity, value, or date rather than a
paragraph. Resolve multi-step questions by following the displayed local
relations, and do not add facts absent from the source text.
{temporal_instruction}
Do not answer "not specified" when a
supporting source cell is present; extract the supported value instead. Return ONLY JSON:
{{"reasoning": "brief source-grounded explanation", "answer": "concise answer"}}
"""
