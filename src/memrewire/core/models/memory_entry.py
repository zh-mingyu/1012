"""Data structures for an addressable source-cell memory substrate."""

from typing import List, Optional
import uuid

from pydantic import BaseModel, Field


class SourceTurn(BaseModel):
    """One original dialogue turn in an interaction-cell ledger."""

    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    source_turn_ids: List[str] = Field(default_factory=list)
    ordinal: int = 0
    source_ordinal: int = 0
    timestamp: Optional[str] = None
    speakers: List[str] = Field(default_factory=list)
    # Nodes are embedded/scored only after their parent cell has been routed.
    embedding: List[float] = Field(default_factory=list)


class CellLink(BaseModel):
    """A persistent edge for local traversal between interaction cells."""

    target_cell_id: str
    relation: str
    weight: float = 0.0
    anchors: List[str] = Field(default_factory=list)


class MemoryCell(BaseModel):
    """Compatibility cell addressed by its routing synopsis."""

    cell_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    head: str
    session_id: Optional[str] = None
    ordinal: int = 0
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    nodes: List[SourceTurn] = Field(default_factory=list)
    links: List[CellLink] = Field(default_factory=list)
    embedding: List[float] = Field(default_factory=list)

    @property
    def routing_text(self) -> str:
        """Return the cell's initial source content for locator encoding.

        The canonical method uses the document encoder on the original cell
        content. Optional maintenance-model addresses remain stored as
        metadata, but they are not mixed into the source locator vector.
        """
        parts = [
            node.text.strip()
            for node in sorted(self.nodes, key=lambda item: (item.ordinal, item.node_id))
            if node.text.strip()
        ]
        content = "\n".join(parts)
        return (content or self.head.strip())[:6000]


class Dialogue(BaseModel):
    """Original dialogue turn with dataset-level provenance."""

    dialogue_id: int
    speaker: str
    content: str
    timestamp: Optional[str] = None
    source_id: Optional[str] = None
    session_id: Optional[str] = None
    turn_index: Optional[int] = None

    @property
    def source_turn_id(self) -> str:
        return str(self.source_id or self.dialogue_id)

    def __str__(self) -> str:
        time_str = f"[{self.timestamp}] " if self.timestamp else ""
        return f"{time_str}{self.speaker}: {self.content}"


class MemoryEntry(BaseModel):
    """Compatibility view for answer/evaluator integrations.

    The cell path does not use this flat object as a persistent index; it is
    created only when a caller requests a flat compatibility result.
    """

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    lossless_restatement: str
    timestamp: Optional[str] = None
    source_turn_ids: List[str] = Field(default_factory=list)
    cell_id: Optional[str] = None
