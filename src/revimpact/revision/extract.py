"""Revision extraction contracts and a conservative pilot extractor."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from revimpact.data.schema import Condition, Revision


def parse_revision_json(text: str) -> Revision:
    """Parse an LLM JSON response after removing an optional markdown fence."""
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S).strip()
    return Revision.model_validate(json.loads(cleaned))


def extract_revision(old_text: str, new_text: str, source_id: Optional[str] = None) -> Revision:
    """Extract simple numeric threshold edits for the dependency-free pilot.

    Real-data runs should replace this with a constrained LLM extractor and
    validate its output through ``parse_revision_json``.
    """
    number = r"(-?\d+(?:\.\d+)?)"
    old_match = re.search(r"(?:上限|limit|maximum)[^\d-]*" + number, old_text, re.I)
    new_match = re.search(r"(?:上限|limit|maximum)[^\d-]*" + number, new_text, re.I)
    if old_match and new_match:
        old_value, new_value = float(old_match.group(1)), float(new_match.group(1))
        revision_type = "numeric_change" if old_value != new_value else "unchanged"
        return Revision(
            predicate="threshold",
            old_value=old_value,
            new_value=new_value,
            condition=Condition(),
            revision_type=revision_type,
            source_id=source_id,
        )
    if old_text == new_text:
        return Revision(predicate="unknown", revision_type="unchanged", source_id=source_id)
    return Revision(predicate="document_content", revision_type="text_change", source_id=source_id)
