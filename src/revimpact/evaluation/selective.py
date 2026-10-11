"""Decision policies and dev-only calibration for selective maintenance."""

from __future__ import annotations

import math

from revimpact.evaluation.metrics import evaluate_records


DECISIONS = ("change", "keep", "abstain")


def apply_policy(scores, policy):
    """Apply a fixed policy to model scores without consulting target fields."""
    predictions = []
    for score in scores:
        mode = policy["mode"]
        if mode == "threshold":
            abstain = score["abstain_confidence"] >= policy["abstain_threshold"]
            change = score["impact_confidence"] >= policy["impact_threshold"]
            decision = "abstain" if abstain else ("change" if change else "keep")
        elif mode in {"head", "calibrated_head"}:
            probabilities = score.get("decision_probabilities")
            if probabilities is None or len(probabilities) != 3:
                raise ValueError("head policy requires a trained three-class decision head")
            if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities):
                raise ValueError("invalid decision probabilities")
            if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
                raise ValueError("decision probabilities must sum to one")
            # Bias correction changes the operating point, not confidence calibration.
            biases = (0.0, policy.get("keep_bias", 0.0), policy.get("abstain_bias", 0.0))
            adjusted = [math.log(max(value, 1e-12)) + bias
                        for value, bias in zip(probabilities, biases)]
            decision = DECISIONS[max(range(3), key=adjusted.__getitem__)]
        else:
            raise ValueError(f"unknown decision policy: {mode}")
        predictions.append({**score, "pred_decision": decision,
                            "pred_should_change": decision == "change",
                            "pred_action": {"change": "update", "keep": "keep", "abstain": "human_review"}[decision]})
    return predictions


def select_policy(dev_rows, scores, policies, minimum_changed_recall=0.95):
    """Pick an operating point on dev, recording an unmet recall constraint."""
    candidates = []
    for policy in policies:
        result = evaluate_records(dev_rows, apply_policy(scores, policy))
        candidates.append({"policy": policy, "changed_recall": result["changed_recall"],
                           "keep_accuracy": result["selective"]["keep_accuracy"],
                           "macro_f1": result["selective"]["macro_f1"],
                           "coverage": result["selective"]["coverage"],
                           "risk": result["selective"]["risk"],
                           "eligible": result["changed_recall"] >= minimum_changed_recall})
    if not candidates:
        raise ValueError("at least one policy is required")
    feasible = [candidate for candidate in candidates if candidate["eligible"]]
    key = lambda candidate: (candidate["macro_f1"], candidate["keep_accuracy"], candidate["coverage"])
    selected = (max(feasible, key=key) if feasible else
                max(candidates, key=lambda candidate: (candidate["changed_recall"], *key(candidate))))
    return {"selected": selected, "candidates": candidates,
            "selection_source": "dev", "minimum_changed_recall": minimum_changed_recall,
            "constraint_satisfied": bool(feasible),
            "selection_reason": ("maximize selective macro-F1 subject to changed recall" if feasible
                                 else "no feasible policy; maximize changed recall first")}


def policy_grid(mode):
    if mode == "threshold":
        return [{"mode": mode, "impact_threshold": round(0.10 + 0.05 * i, 2),
                 "abstain_threshold": round(0.30 + 0.05 * j, 2)}
                for i in range(17) for j in range(15)]
    if mode == "head":
        return [{"mode": mode, "keep_bias": 0.0, "abstain_bias": 0.0}]
    if mode == "calibrated_head":
        return [{"mode": mode, "keep_bias": round(0.10 * i, 2),
                 "abstain_bias": round(0.25 * j, 2)}
                for i in range(-20, 21) for j in range(-8, 9)]
    raise ValueError(f"unknown decision policy: {mode}")
