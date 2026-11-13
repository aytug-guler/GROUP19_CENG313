"""Shared behavior metadata and submission helpers."""

import ast
import re

import pandas as pd


SUBMISSION_COLUMNS = [
    "row_id", "video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame",
]


def mouse_name(value):
    """Normalize numeric and mouse-prefixed identifiers."""
    match = re.fullmatch(r"(?:mouse)?([1-9][0-9]*)(?:\\.0+)?", str(value).strip())
    return f"mouse{match[1]}" if match else None


def parse_behaviors(value):
    """Return the annotated (agent, target, action) triples for one video."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return set()
    if not isinstance(value, (list, tuple)):
        return set()

    behaviors = set()
    for label in value:
        if not isinstance(label, str):
            continue
        parts = [part.strip() for part in label.split(",")]
        if len(parts) != 3:
            continue
        raw_agent, raw_target, action = parts
        agent = mouse_name(raw_agent)
        target = "self" if raw_target == "self" else mouse_name(raw_target)
        if agent and target and action:
            behaviors.add((agent, "self" if target == agent else target, action))
    return behaviors


def save_submission(events, path):
    """Number validated events and write them in Kaggle column order."""
    submission = pd.DataFrame(events, columns=SUBMISSION_COLUMNS[1:])
    submission.insert(0, "row_id", range(len(submission)))
    submission.to_csv(path, index=False)
    return submission


def consecutive_intervals(frames):
    """Merge sorted positive frames into half-open intervals."""
    stop = None
    for frame in frames:
        frame = int(frame)
        if stop is None:
            start = frame
        elif frame != stop:
            yield start, stop
            start = frame
        stop = frame + 1
    if len(frames):
        yield start, stop
