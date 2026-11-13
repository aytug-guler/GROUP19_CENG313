
import numpy as np
import pandas as pd
from pathlib import Path

from .utils import mouse_name


FEATURE_COLUMNS = [
    "nose_to_neck_cm",
    "nose_to_nose_cm",
    "nose_to_tail_cm",
    "neck_to_neck_cm",
    "mouse1_speed",
    "mouse2_speed",
    "distance_change",
]


def extract_features(tracking_df, pixels_per_cm, fps):
    """Extract tracking-based features for two mice."""

    if pixels_per_cm <= 0 or fps <= 0:
        raise ValueError(
            "FPS and pixels_per_cm must be positive."
        )

    selected_parts = ["nose", "neck", "tail_base"]

    subset = tracking_df[
        tracking_df["bodypart"].isin(selected_parts)
        & tracking_df["mouse_id"].isin([1, 2])
    ].copy()

    # Treat (0, 0) points as invalid measurements
    invalid = (
        (subset["x"] == 0) & (subset["y"] == 0)
    )
    subset.loc[invalid, ["x", "y"]] = np.nan

    # Convert tracking data into wide format
    positions = subset.pivot(
        index="video_frame",
        columns=["mouse_id", "bodypart"],
        values=["x", "y"],
    ).sort_index()

    # Preserve all original frame numbers
    positions = positions.reindex(
        np.sort(tracking_df["video_frame"].unique())
    )

    def point(mouse, part):
        return (
            positions[("x", mouse, part)],
            positions[("y", mouse, part)],
        )

    def distance(p1, p2):
        return np.sqrt(
            (p1[0] - p2[0]) ** 2
            + (p1[1] - p2[1]) ** 2
        ) / pixels_per_cm

    m1_nose = point(1, "nose")
    m1_neck = point(1, "neck")

    m2_nose = point(2, "nose")
    m2_neck = point(2, "neck")
    m2_tail = point(2, "tail_base")

    features = pd.DataFrame(index=positions.index)

    # Relative distances in centimeters
    features["nose_to_neck_cm"] = distance(
        m1_nose, m2_neck
    )

    features["nose_to_nose_cm"] = distance(
        m1_nose, m2_nose
    )

    features["nose_to_tail_cm"] = distance(
        m1_nose, m2_tail
    )

    features["neck_to_neck_cm"] = distance(
        m1_neck, m2_neck
    )

    # Frame number differences
    frame_delta = pd.Series(
        positions.index,
        index=positions.index,
    ).diff()

    # Movement speeds in cm/s
    for mouse in [1, 2]:
        x, y = point(mouse, "nose")

        displacement = np.sqrt(
            x.diff() ** 2 + y.diff() ** 2
        )

        features[f"mouse{mouse}_speed"] = (
            displacement * fps
            / (pixels_per_cm * frame_delta)
        )

    # Change in inter-mouse distance per frame
    features["distance_change"] = (
        features["nose_to_neck_cm"].diff()
        / frame_delta
    )

    return features.replace(
        [np.inf, -np.inf], np.nan
    )


def process_video(
    video_id, eligible_metadata, data_dir
):
    """Prepare frame-level features and binary labels."""

    video_id = int(video_id)
    lab = "CalMS21_task1"
    data_dir = Path(data_dir)

    # Find metadata for the selected video
    row = eligible_metadata[
        eligible_metadata["video_id"] == video_id
    ].iloc[0]

    tracking_path = (
        data_dir / "train_tracking"
        / lab / f"{video_id}.parquet"
    )

    annotation_path = (
        data_dir / "train_annotation"
        / lab / f"{video_id}.parquet"
    )

    # Load tracking and annotation files
    tracking_df = pd.read_parquet(tracking_path)
    annotations_df = pd.read_parquet(annotation_path)

    # Extract numerical features
    features = extract_features(
        tracking_df,
        row["pix_per_cm_approx"],
        row["frames_per_second"],
    )

    # Create binary frame-level labels
    frames = features.index.to_numpy()
    labels = np.zeros(len(frames), dtype=np.int8)

    # Mouse 1 -> Mouse 2 sniffbody events
    events = annotations_df[
        (annotations_df["action"] == "sniffbody")
        & (annotations_df["agent_id"] == 1)
        & (annotations_df["target_id"] == 2)
    ]

    # Apply half-open annotation intervals
    for event in events.itertuples():
        labels[
            (frames >= event.start_frame)
            & (frames < event.stop_frame)
        ] = 1

    result = features.copy()

    result["label"] = labels
    result["video_id"] = video_id

    # Remove frames with incomplete features
    result = result.dropna(
        subset=FEATURE_COLUMNS
    )

    return result.reset_index()


def build_split(
    video_ids, split_name, eligible_metadata, data_dir
):
    """Build a dataset from a video-level split."""

    parts = []

    for i, video_id in enumerate(video_ids, 1):
        part = process_video(
            video_id,
            eligible_metadata,
            data_dir,
        )

        parts.append(part)

        if i % 10 == 0 or i == len(video_ids):
            print(
                f"{split_name}: "
                f"{i}/{len(video_ids)} videos processed"
            )

    if not parts:
        raise ValueError("No videos provided.")

    dataset = pd.concat(parts, ignore_index=True)

    print(f"\n{split_name} completed")

    print(
        "Total valid frames:",
        len(dataset)
    )

    print(
        "Positive frames:",
        int(dataset["label"].sum())
    )

    print(
        "Negative frames:",
        int((dataset["label"] == 0).sum())
    )

    print(
        "Positive ratio:",
        round(dataset["label"].mean() * 100, 2),
        "%"
    )

    return dataset


# Shared features for the competition models. The historical features above
# remain available for reproducing the original sniffbody experiment.
TRACK_COLUMNS = ["video_frame", "mouse_id", "bodypart", "x", "y"]
PAIR_FEATURES = ["agent_speed", "target_speed", "distance", "closing_speed", "self_target"]


def video_features(tracking, scale, fps):
    """Pool available keypoints into per-mouse centers and estimate speed."""
    scale = scale if pd.notna(scale) and np.isfinite(scale) and scale > 0 else 1.0
    fps = fps if pd.notna(fps) and np.isfinite(fps) and fps > 0 else 30.0
    points = tracking.copy()
    names = {value: mouse_name(value) for value in points["mouse_id"].dropna().unique()}
    points["mouse_id"] = points["mouse_id"].map(names)
    frames = pd.to_numeric(points["video_frame"], errors="coerce")
    valid = np.isfinite(frames) & frames.ge(0) & frames.mod(1).eq(0)
    points = points.loc[valid].copy()
    points["video_frame"] = frames.loc[valid].astype("int64")
    xy = points[["x", "y"]].apply(pd.to_numeric, errors="coerce")
    points[["x", "y"]] = xy.where(np.isfinite(xy).all(axis=1) & xy.ne(0).any(axis=1))
    points = points.dropna(subset=["mouse_id", "x", "y"]).drop_duplicates(
        ["video_frame", "mouse_id", "bodypart"]
    )
    centers = points.groupby(["mouse_id", "video_frame"])[["x", "y"]].mean() / scale
    mice = {}
    for mouse, center in centers.groupby(level="mouse_id"):
        center = center.droplevel("mouse_id").sort_index()
        delta = center.index.to_series().diff()
        speed = np.hypot(center["x"].diff(), center["y"].diff()) * fps / delta
        mice[mouse] = center.assign(speed=speed.fillna(0.0))
    return mice


def pair_features(mice, agent, target, fps):
    """Represent a directed pair or a self action on jointly observed frames."""
    if target == "self":
        target = agent
    first, second = mice[agent].align(mice[target], join="inner", axis=0)
    distance = np.hypot(first["x"] - second["x"], first["y"] - second["y"])
    fps = fps if pd.notna(fps) and np.isfinite(fps) and fps > 0 else 30.0
    closing = -distance.diff() * fps / distance.index.to_series().diff()
    return pd.DataFrame({
        "agent_speed": first["speed"], "target_speed": second["speed"],
        "distance": distance, "closing_speed": closing.fillna(0.0),
        "self_target": float(agent == target),
    }, index=first.index).astype("float32")


def annotation_events(path):
    """Group half-open intervals by normalized agent, target and action."""
    groups = {}
    for agent, target, action, start, stop in pd.read_parquet(path)[
        ["agent_id", "target_id", "action", "start_frame", "stop_frame"]
    ].itertuples(index=False, name=None):
        agent = mouse_name(agent)
        target = "self" if str(target) == "self" else mouse_name(target)
        if agent and target and pd.notna(action) and pd.notna(start) and pd.notna(stop) and 0 <= start < stop:
            groups.setdefault((agent, "self" if target == agent else target, action), []).append(
                (int(start), int(stop))
            )
    return groups


def interval_labels(frames, intervals):
    """Label observed frames without expanding unobserved video ranges."""
    changes = np.zeros(len(frames) + 1, dtype=np.int32)
    for start, stop in intervals:
        left, right = np.searchsorted(frames, [start, stop])
        changes[left] += 1
        changes[right] -= 1
    return changes[:-1].cumsum() > 0
