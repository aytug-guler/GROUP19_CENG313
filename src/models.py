
"""
Baseline models for CENG313 - MABe Challenge.

This module contains reusable functions for training
and evaluating frame-level binary behavior classifiers.
"""

import numpy as np
import pandas as pd

from collections import Counter, defaultdict
from pathlib import Path

from .data import TRACK_COLUMNS, annotation_events, interval_labels, pair_features, video_features
from .utils import SUBMISSION_COLUMNS, consecutive_intervals, parse_behaviors, save_submission

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)


def train_dummy_model(X_train, y_train):
    """Train a majority-class Dummy Classifier."""

    model = DummyClassifier(
        strategy="most_frequent"
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):
    """Train the Assignment 1 Random Forest baseline."""

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X, y, threshold=0.5):
    """
    Evaluate a binary classifier at a fixed threshold.

    Returns evaluation metrics and predictions.
    """

    class_probabilities = model.predict_proba(X)
    positive = np.flatnonzero(model.classes_ == 1)
    probabilities = (
        class_probabilities[:, positive[0]] if positive.size else np.zeros(len(X))
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y, predictions, zero_division=0
        ),
        "Average Precision": average_precision_score(
            y, probabilities
        ),
    }

    matrix = confusion_matrix(
        y, predictions, labels=[0, 1]
    )

    return {
        "metrics": metrics,
        "predictions": predictions,
        "probabilities": probabilities,
        "confusion_matrix": matrix,
    }


def find_best_threshold(y_true, probabilities):
    """
    Find the threshold maximizing validation F1-score.

    This function should only be used on validation data,
    never on the independent test set.
    """

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true, probabilities
        )
    )

    if len(thresholds) == 0:
        raise ValueError(
            "No valid decision thresholds found."
        )

    precision_t = precision[:-1]
    recall_t = recall[:-1]

    f1_values = (
        2 * precision_t * recall_t
        / (precision_t + recall_t + 1e-12)
    )

    best_index = int(np.argmax(f1_values))

    best_threshold = float(
        thresholds[best_index]
    )

    return {
        "threshold": best_threshold,
        "f1_score": float(f1_values[best_index]),
        "thresholds": thresholds,
        "f1_values": f1_values,
    }


def get_feature_importance(model, feature_names):
    """
    Return feature importance values in descending order.
    """

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    })

    return importance.sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)


def annotated_videos(metadata, root):
    """Keep videos with both inputs needed for supervised learning."""
    records = []
    for row in metadata.itertuples(index=False):
        filename = f"{row.video_id}.parquet"
        if all((root / folder / row.lab_id / filename).is_file()
               for folder in ("train_tracking", "train_annotation")):
            records.append(row)
    return records


def split_behavior_videos(records):
    """Hold out approximately 20% of videos per lab, keeping singleton labs for training."""
    by_lab = defaultdict(list)
    for row in records:
        by_lab[row.lab_id].append(row)
    rng = np.random.default_rng(42)
    fit, heldout = [], []
    for rows in by_lab.values():
        rows.sort(key=lambda row: row.video_id)
        count = max(1, round(0.2 * len(rows))) if len(rows) > 1 else 0
        for index, position in enumerate(rng.permutation(len(rows))):
            (heldout if index < count else fit).append(rows[position])
    return fit, heldout


def read_video_mice(root, folder, row):
    path = root / folder / row.lab_id / f"{row.video_id}.parquet"
    tracking = pd.read_parquet(path, columns=TRACK_COLUMNS)
    return video_features(tracking, row.pix_per_cm_approx, row.frames_per_second)


def allowed_pairs(row, mice, actions):
    pairs = defaultdict(list)
    for agent, target, action in sorted(parse_behaviors(row.behaviors_labeled)):
        if action in actions and agent in mice and (target == "self" or target in mice):
            pairs[agent, target].append(action)
    return pairs


def train_behavior_models(records, root, actions):
    """Fit one forest per action using bounded positive/negative samples per video."""
    scopes = [
        {label for label in parse_behaviors(row.behaviors_labeled) if label[2] in actions}
        for row in records
    ]
    counts = Counter(label[2] for scope in scopes for label in scope)
    rng = np.random.default_rng(42)
    samples = defaultdict(list)
    for row, scope in zip(records, scopes):
        if not scope:
            continue
        mice = read_video_mice(root, "train_tracking", row)
        events = annotation_events(root / "train_annotation" / row.lab_id / f"{row.video_id}.parquet")
        for (agent, target), pair_actions in allowed_pairs(row, mice, actions).items():
            features = pair_features(mice, agent, target, row.frames_per_second)
            for action in pair_actions:
                labels = interval_labels(features.index.to_numpy(), events.get((agent, target, action), []))
                quota = max(4, min(400, 10000 // counts[action]))
                for positive in (False, True):
                    candidates = np.flatnonzero(labels == positive)
                    if len(candidates):
                        indices = rng.choice(candidates, min(quota, len(candidates)), replace=False)
                        samples[action].append(features.iloc[indices].assign(label=int(positive)))
    forests = {}
    for action, batches in samples.items():
        training = pd.concat(batches, ignore_index=True)
        if training["label"].nunique() == 2:
            forests[action] = train_random_forest(training.drop(columns="label"), training["label"])
    return forests


def predict_behaviors(row, mice, models):
    """Choose at most one action per pair and frame at a fixed 0.5 threshold."""
    events = []
    for (agent, target), actions in allowed_pairs(row, mice, models).items():
        features = pair_features(mice, agent, target, row.frames_per_second)
        if features.empty:
            continue
        probabilities = np.column_stack([models[action].predict_proba(features)[:, 1] for action in actions])
        winner = probabilities.argmax(axis=1)
        confident = probabilities.max(axis=1) >= 0.5
        frames = features.index.to_numpy()
        for index, action in enumerate(actions):
            events.extend(
                (agent, target, action, start, stop)
                for start, stop in consecutive_intervals(frames[confident & (winner == index)])
            )
    return events


def evaluate_behaviors(records, root, actions, models):
    """Micro frame F1 per action over observed frames in held-out labeled pairs."""
    counts = {action: np.zeros(3, dtype=np.int64) for action in sorted(actions)}
    for row in records:
        if not any(label[2] in actions for label in parse_behaviors(row.behaviors_labeled)):
            continue
        mice = read_video_mice(root, "train_tracking", row)
        truth = annotation_events(root / "train_annotation" / row.lab_id / f"{row.video_id}.parquet")
        predicted = defaultdict(list)
        for agent, target, action, start, stop in predict_behaviors(row, mice, models):
            predicted[agent, target, action].append((start, stop))
        for (agent, target), pair_actions in allowed_pairs(row, mice, actions).items():
            frames = pair_features(mice, agent, target, row.frames_per_second).index.to_numpy()
            for action in pair_actions:
                key = agent, target, action
                actual = interval_labels(frames, truth.get(key, []))
                result = interval_labels(frames, predicted[key])
                counts[action] += [(actual & result).sum(), actual.sum(), result.sum()]
    return pd.DataFrame([
        (action, action in models, actual, predicted, 2 * tp / (actual + predicted) if actual + predicted else np.nan)
        for action, (tp, actual, predicted) in counts.items()
    ], columns=["action", "trained", "positive_frames", "predicted_frames", "frame_f1"])


def generate_submission(root, output):
    """Train from public annotations, infer on attached test files, and validate CSV."""
    root, output = Path(root), Path(output)
    train, test = (pd.read_csv(root / name) for name in ("train.csv", "test.csv"))
    if test["video_id"].duplicated().any():
        raise ValueError("Duplicate test video IDs")
    if pd.read_csv(root / "sample_submission.csv", nrows=0).columns.tolist() != SUBMISSION_COLUMNS:
        raise ValueError("Unexpected competition CSV header")
    actions = {label[2] for value in test["behaviors_labeled"] for label in parse_behaviors(value)}
    forests = train_behavior_models(annotated_videos(train, root), root, actions)
    if not forests:
        raise RuntimeError("No requested action has both positive and negative annotated training frames.")
    print("Trained actions:", sorted(forests), "| No two-class training support:", sorted(actions - forests.keys()))
    events = []
    for row in test.itertuples(index=False):
        mice = read_video_mice(root, "test_tracking", row)
        allowed = parse_behaviors(row.behaviors_labeled)
        predictions = predict_behaviors(row, mice, forests)
        previous_stop = {}
        frame_sets = {}
        for agent, target, action, start, stop in sorted(predictions, key=lambda event: (event[0], event[1], event[3])):
            pair = agent, target
            if pair not in frame_sets:
                frame_sets[pair] = pair_features(mice, agent, target, row.frames_per_second).index.to_numpy()
            frames = frame_sets[pair]
            left, right = np.searchsorted(frames, [start, stop])
            if ((agent, target, action) not in allowed or not 0 <= start < stop
                    or right - left != stop - start or start < previous_stop.get(pair, 0)):
                raise ValueError(f"Invalid prediction for video {row.video_id}")
            previous_stop[pair] = stop
            events.append((int(row.video_id), agent, target, action, start, stop))
    output.parent.mkdir(parents=True, exist_ok=True)
    submission = save_submission(events, output)
    saved = pd.read_csv(output)
    if (saved.columns.tolist() != SUBMISSION_COLUMNS or saved.isna().any().any()
            or not np.array_equal(saved["row_id"], np.arange(len(saved)))
            or (len(saved) and not all(pd.api.types.is_integer_dtype(saved[column])
                                       for column in ("row_id", "video_id", "start_frame", "stop_frame")))):
        raise ValueError("Saved submission failed schema validation")
    print(f"Validated {len(saved):,} intervals for {saved.video_id.nunique()} of {len(test)} test videos.")
    return submission
