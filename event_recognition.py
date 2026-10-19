"""Football event inference adapter and goal-to-reaction linker.

The adapter intentionally does not pretend generic audio peaks are semantic labels.
A validated model is loaded through PITCHPERFECT_EVENT_MODEL_COMMAND, an executable
that accepts --video PATH --output PATH and writes a JSON list of predictions:
[{"timestamp": 123.4, "label": "goal", "confidence": 0.91}].
See README for the schema and validation workflow.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SUPPORTED_LABELS = {
    "goal", "celebration", "emotional_reaction", "manager_reaction",
    "shot_on_target", "penalty", "save", "crowd_reaction", "unknown",
}


@dataclass(frozen=True)
class EventPrediction:
    timestamp: float
    label: str
    confidence: float
    source: str = "football_model"
    evidence: str | None = None


def run_event_model(video_path: Path, duration: float) -> tuple[list[EventPrediction], dict[str, Any]]:
    """Run the configured model adapter. No model configured means no semantic predictions."""
    command_template = os.getenv("PITCHPERFECT_EVENT_MODEL_COMMAND", "").strip()
    if not command_template:
        return [], {
            "configured": False,
            "model_name": None,
            "status": "not_configured",
            "message": "No football event model configured; audio peaks remain review-only candidates.",
        }

    output_path = video_path.with_name(f"{video_path.stem}.event_predictions.json")
    has_video = "{video}" in command_template
    has_output = "{output}" in command_template
    command = command_template.format(video=str(video_path), output=str(output_path))
    argv = shlex.split(command)
    if not has_video:
        argv.extend(["--video", str(video_path)])
    if not has_output:
        argv.extend(["--output", str(output_path)])
    try:
        completed = subprocess.run(
            argv, check=False, capture_output=True, text=True, timeout=7200
        )
        if completed.returncode != 0:
            return [], {"configured": True, "status": "failed", "message": completed.stderr[-1200:]}
        if not output_path.exists():
            return [], {"configured": True, "status": "failed", "message": "Model command did not create its output JSON."}
        raw = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Prediction JSON must be a list.")
        predictions: list[EventPrediction] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "unknown")).lower().strip()
            try:
                timestamp = float(item["timestamp"])
                confidence = float(item["confidence"])
            except (KeyError, TypeError, ValueError):
                continue
            if label not in SUPPORTED_LABELS or not (0 <= timestamp <= duration):
                continue
            if not (0.0 <= confidence <= 1.0):
                continue
            predictions.append(EventPrediction(
                timestamp=timestamp, label=label, confidence=confidence,
                source=str(item.get("source", "football_model")),
                evidence=str(item["evidence"])[:500] if item.get("evidence") else None,
            ))
        return sorted(predictions, key=lambda p: p.timestamp), {
            "configured": True,
            "model_name": os.getenv("PITCHPERFECT_EVENT_MODEL_NAME", "external-football-event-model"),
            "status": "ok",
            "prediction_count": len(predictions),
        }
    except Exception as exc:
        return [], {"configured": True, "status": "failed", "message": str(exc)[:1200]}
    finally:
        output_path.unlink(missing_ok=True)


def link_goal_sequences(predictions: list[EventPrediction], duration: float,
                        before: float = 12.0, after: float = 35.0) -> list[dict[str, Any]]:
    """Link predicted goals with nearby celebration/reaction events; never invent missing labels."""
    goals = [p for p in predictions if p.label == "goal"]
    linked: list[dict[str, Any]] = []
    reaction_labels = {"celebration", "emotional_reaction", "manager_reaction", "crowd_reaction"}
    for index, goal in enumerate(goals, start=1):
        related = [p for p in predictions if p.label in reaction_labels
                   and goal.timestamp - before <= p.timestamp <= goal.timestamp + after]
        linked.append({
            "event_id": f"auto-goal-{index}-{int(goal.timestamp * 1000)}",
            "goal": asdict(goal),
            "linked_events": [asdict(p) for p in sorted(related, key=lambda p: p.timestamp)],
            "window": {"start_seconds": max(0.0, goal.timestamp - before),
                       "end_seconds": min(duration, goal.timestamp + after)},
            "linking_method": "nearest-temporal-context",
            "review_required": True,
        })
    return linked
