"""Run a pretrained OpenSportsLib/SoccerNet action-spotting model on a match.

This adapter intentionally emits only labels that the selected checkpoint is
trained to predict. The current configured checkpoint includes GOAL, but not
celebration, emotional_reaction, or manager_reaction.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_WEIGHTS = "OpenSportsLab/OSL-loc-snbas-2025-e2e"


def _walk_events(payload: Any):
    """Yield event dicts from the documented OpenSportsLib prediction payload."""
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            for sample in data:
                if not isinstance(sample, dict):
                    continue
                events = sample.get("events", [])
                if isinstance(events, list):
                    for event in events:
                        if isinstance(event, dict):
                            yield event
        # Some wrappers return one sample or an event list directly.
        events = payload.get("events")
        if isinstance(events, list):
            for event in events:
                if isinstance(event, dict):
                    yield event
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                if "events" in item and isinstance(item["events"], list):
                    for event in item["events"]:
                        if isinstance(event, dict):
                            yield event
                elif "label" in item:
                    yield item


def convert_predictions(payload: Any, duration_seconds: float,
                        threshold: float = 0.35) -> list[dict[str, Any]]:
    """Convert OSL spotting output into PitchPerfect's seconds-based schema."""
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for event in _walk_events(payload):
        raw_label = str(event.get("label", "")).strip().lower().replace("_", " ")
        # The configured public checkpoint is trained for GOAL among its classes.
        if raw_label not in {"goal", "goals"}:
            continue
        position_ms = event.get("position_ms")
        if position_ms is not None:
            try:
                timestamp = float(position_ms) / 1000.0
            except (TypeError, ValueError):
                continue
        elif event.get("timestamp") is not None:
            try:
                timestamp = float(event["timestamp"])
            except (TypeError, ValueError):
                continue
        else:
            continue
        try:
            confidence = float(event.get("confidence", event.get("confidence_score", 0.0)))
        except (TypeError, ValueError):
            continue
        if not (0 <= timestamp <= duration_seconds and threshold <= confidence <= 1):
            continue
        key = ("goal", round(timestamp * 1000))
        if key in seen:
            continue
        seen.add(key)
        output.append({
            "timestamp": timestamp,
            "label": "goal",
            "confidence": confidence,
            "source": "opensportslib-soccernet-pretrained",
            "evidence": "Pretrained SoccerNet ball-action spotting prediction; verify against broadcast/replay context.",
        })
    return sorted(output, key=lambda item: item["timestamp"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights", default=os.getenv("PITCHPERFECT_SOCCERNET_WEIGHTS", DEFAULT_WEIGHTS))
    parser.add_argument("--threshold", type=float, default=float(os.getenv("PITCHPERFECT_GOAL_THRESHOLD", "0.35")))
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"Video does not exist: {args.video}")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    # Import lazily so the regular API can run without the optional model stack.
    try:
        from huggingface_hub import hf_hub_download
        from opensportslib.apis import LocalizationModel
    except ImportError as exc:
        raise SystemExit(
            "OpenSportsLib / Hugging Face Hub is not installed. Install the optional "
            "requirements with `python -m pip install -r requirements-event-model.txt`."
        ) from exc

    print(f"Resolving model config for checkpoint: {args.weights}", flush=True)
    try:
        config_path = hf_hub_download(repo_id=args.weights, filename="config.yaml")
    except Exception as exc:
        raise SystemExit(
            f"Could not download config.yaml from model repository {args.weights!r}. "
            "Check internet access, Hugging Face availability, and the repository ID. "
            f"Details: {exc}"
        ) from exc

    print(f"Loading SoccerNet action-spotting checkpoint: {args.weights}", flush=True)
    # The published model card documents both a config path and a weights repository ID.
    model = LocalizationModel(config=str(config_path), weights=args.weights)
    print(f"Running inference on {args.video}", flush=True)
    predictions = model.infer(video_path=str(args.video))

    # Duration is derived from OpenCV metadata; avoid accepting timestamps outside it.
    import cv2
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    duration = frames / fps if fps > 0 else float("inf")
    converted = convert_predictions(predictions, duration, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, indent=2), encoding="utf-8")
    print(f"Wrote {len(converted)} goal candidates to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
