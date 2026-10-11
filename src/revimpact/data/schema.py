"""Stable JSONL schema shared by collectors, models, and evaluators."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Conclusion(str, Enum):
    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class Action(str, Enum):
    KEEP = "keep"
    VERIFY = "verify"
    UPDATE = "update"
    INVALIDATE = "invalidate"
    HUMAN_REVIEW = "human_review"


class Condition(BaseModel):
    """Normalized query or revision conditions.

    Values are deliberately JSON-friendly. Numeric values may be represented as
    a scalar or as an interval ``{"min": ..., "max": ...}``.
    """

    slots: Dict[str, Any] = Field(default_factory=dict)
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None


class EvidenceSpan(BaseModel):
    version: str
    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    source_id: Optional[str] = None


class Revision(BaseModel):
    predicate: str
    old_value: Any = None
    new_value: Any = None
    condition: Condition = Field(default_factory=Condition)
    revision_type: str = "unknown"
    source_id: Optional[str] = None


class RevisionRecord(BaseModel):
    """One evaluation instance.

    ``family_id`` is the unit used for document-family splits. The explicit
    gold fields make leakage and annotation audits possible.
    """

    id: str
    family_id: str
    domain: str
    old_version: str
    new_version: str
    old_document: str
    new_document: str
    query: str
    condition: Condition = Field(default_factory=Condition)
    revision: Revision
    old_label: Conclusion
    new_label: Conclusion
    should_change: bool
    impact_condition: Condition = Field(default_factory=Condition)
    old_evidence: List[EvidenceSpan] = Field(default_factory=list)
    new_evidence: List[EvidenceSpan] = Field(default_factory=list)
    action: Action
    revision_type: str = "unknown"
    split: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("should_change")
    @classmethod
    def validate_change(cls, value: bool) -> bool:
        if not isinstance(value, bool):
            raise TypeError("should_change must be boolean")
        return value

    def to_json_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, value: Dict[str, Any]) -> "RevisionRecord":
        return cls.model_validate(value)
