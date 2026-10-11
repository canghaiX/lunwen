"""Shared prompts, conservative JSON parsing, and annotation helpers."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Mapping

from revimpact.data.schema import Action, Conclusion


LABELS = {item.value for item in Conclusion}
ACTIONS = {item.value for item in Action}


def public_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields an annotator is allowed to see."""
    return {
        "id": row["id"], "family_id": row.get("family_id"), "domain": row.get("domain"),
        "old_version": row.get("old_version"), "new_version": row.get("new_version"),
        "old_document": row.get("old_document", ""), "new_document": row.get("new_document", ""),
        "query": row.get("query", ""), "condition": row.get("condition", {}),
        "revision": row.get("revision", {}),
        "old_evidence_candidates": row.get("old_evidence", []),
        "new_evidence_candidates": row.get("new_evidence", []),
    }


def initial_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    payload = json.dumps(public_record(row), ensure_ascii=False, sort_keys=True)
    return [
        {"role": "system", "content": "You are a careful document revision annotator. Return only valid JSON, no markdown and no reasoning."},
        {"role": "user", "content": (
            "Annotate this old/new document pair. Decide each version's entailment of the query, "
            "whether the query conclusion truly changes, supporting evidence IDs, applicable impact "
            "condition, and maintenance action. Do not infer a change merely from text edits. "
            "Labels: entailed, contradicted, unknown. Actions: keep, verify, update, invalidate, human_review.\n"
            "JSON schema: {old_label, new_label, should_change, old_evidence_ids, new_evidence_ids, "
            "impact_condition, action, confidence, uncertainty}\nINPUT:\n" + payload
        )},
    ]


def review_messages(row: Mapping[str, Any], initial: Mapping[str, Any]) -> list[dict[str, str]]:
    payload = json.dumps(public_record(row), ensure_ascii=False, sort_keys=True)
    proposal = json.dumps(initial, ensure_ascii=False, sort_keys=True)
    return [
        {"role": "system", "content": "You are an independent senior annotation auditor. Return only valid JSON, no markdown and no reasoning."},
        {"role": "user", "content": (
            "Audit the proposed annotation against the document pair and query. Recompute every field "
            "independently. For each field use decision accept, correct, or uncertain. Do not treat the "
            "proposal as gold.\nReturn JSON with fields: old_label, new_label, should_change, evidence, "
            "action, overall_decision, confidence, reason_codes. Each of old_label/new_label/should_change/action "
            "must contain {decision, value}. evidence may contain {decision, old_evidence_ids, new_evidence_ids}.\n"
            "DOCUMENT INPUT:\n" + payload + "\nPROPOSED ANNOTATION:\n" + proposal
        )},
    ]


def _json_candidate(text: str) -> str | None:
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", flags=re.S | re.I)
    candidates = fenced or re.findall(r"\{.*\}", text or "", flags=re.S)
    for candidate in reversed(candidates):
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    return None


def parse_initial(text: str) -> tuple[dict[str, Any], bool]:
    candidate = _json_candidate(text)
    if candidate is None:
        return {
            "old_label": "unknown", "new_label": "unknown", "should_change": False,
            "old_evidence_ids": [], "new_evidence_ids": [], "impact_condition": {},
            "action": "human_review", "confidence": 0.0, "uncertainty": ["parse_failed"],
        }, True
    value = json.loads(candidate)
    old = value.get("old_label") if value.get("old_label") in LABELS else "unknown"
    new = value.get("new_label") if value.get("new_label") in LABELS else "unknown"
    action = value.get("action") if value.get("action") in ACTIONS else "human_review"
    confidence = value.get("confidence", 0.0)
    try:
        confidence = min(1.0, max(0.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "old_label": old, "new_label": new, "should_change": bool(value.get("should_change", old != new)),
        "old_evidence_ids": [str(x) for x in value.get("old_evidence_ids", []) if x is not None],
        "new_evidence_ids": [str(x) for x in value.get("new_evidence_ids", []) if x is not None],
        "impact_condition": value.get("impact_condition", {}) if isinstance(value.get("impact_condition", {}), dict) else {},
        "action": action, "confidence": confidence,
        "uncertainty": ([str(x) for x in value.get("uncertainty", [])]
                        if isinstance(value.get("uncertainty", []), (list, tuple))
                        else [str(value.get("uncertainty"))] if value.get("uncertainty") else []),
    }, False


def parse_review(text: str) -> tuple[dict[str, Any], bool]:
    candidate = _json_candidate(text)
    if candidate is None:
        return {"overall_decision": "human_review", "confidence": 0.0, "reason_codes": ["parse_failed"]}, True
    value = json.loads(candidate)
    reason_codes = value.get("reason_codes", [])
    if not isinstance(reason_codes, (list, tuple)):
        reason_codes = [reason_codes] if reason_codes else []
    overall = value.get("overall_decision", "human_review")
    if isinstance(overall, dict):
        overall = overall.get("decision", overall.get("value", "human_review"))
    if overall == "correct":
        overall = "revise"
    elif overall not in {"accept", "revise", "human_review"}:
        overall = "human_review"
    normalized: dict[str, Any] = {
        "overall_decision": overall,
        "confidence": value.get("confidence", 0.0),
        "reason_codes": [str(x) for x in reason_codes],
    }
    for field in ("old_label", "new_label", "should_change", "action"):
        item = value.get(field, {})
        if not isinstance(item, dict):
            item = {"decision": "uncertain", "value": item}
        decision = item.get("decision", "uncertain")
        if decision not in {"accept", "correct", "uncertain"}:
            decision = "uncertain"
        if field in {"old_label", "new_label"}:
            val = item.get("value") if item.get("value") in LABELS else "unknown"
        elif field == "action":
            val = item.get("value") if item.get("value") in ACTIONS else "human_review"
        else:
            val = bool(item.get("value", False))
        normalized[field] = {"decision": decision, "value": val}
    evidence = value.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    old_evidence_ids = evidence.get("old_evidence_ids", [])
    new_evidence_ids = evidence.get("new_evidence_ids", [])
    if not isinstance(old_evidence_ids, (list, tuple)):
        old_evidence_ids = [old_evidence_ids] if old_evidence_ids else []
    if not isinstance(new_evidence_ids, (list, tuple)):
        new_evidence_ids = [new_evidence_ids] if new_evidence_ids else []
    normalized["evidence"] = {
        "decision": evidence.get("decision", "uncertain"),
        "old_evidence_ids": [str(x) for x in old_evidence_ids],
        "new_evidence_ids": [str(x) for x in new_evidence_ids],
    }
    try:
        normalized["confidence"] = min(1.0, max(0.0, float(normalized["confidence"])))
    except (TypeError, ValueError):
        normalized["confidence"] = 0.0
    return normalized, False


def openai_chat(base_url: str, messages: list[dict[str, str]], model: str, *, timeout: int = 600, max_tokens: int = 512) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps({
        "model": model, "messages": messages, "temperature": 0,
        "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    return str(value["choices"][0]["message"]["content"])
