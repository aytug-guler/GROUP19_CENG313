
"""
Baseline models for CENG313 - MABe Challenge.

This module contains reusable functions for training
and evaluating frame-level binary behavior classifiers.
"""

import numpy as np
import pandas as pd

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

    probabilities = model.predict_proba(X)[:, 1]

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
