"""
Trajectory interpolation utilities for the UMI recording pipeline.

Handles interpolation from 50Hz trajectory data to 60Hz video timestamps
using linear interpolation for position and SLERP for rotation.
"""

from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import pyarrow as pa
import pyarrow.parquet as pq

from .schemas import (
    TRAJECTORY_SCHEMA,
    TrajectoryFrame,
    VideoTimestamp,
)


def interpolate_position(
    t: float,
    t0: float,
    p0: np.ndarray,
    t1: float,
    p1: np.ndarray,
) -> np.ndarray:
    """
    Linear interpolation of position.

    Args:
        t: Target timestamp
        t0: Before timestamp
        p0: Position at t0
        t1: After timestamp
        p1: Position at t1

    Returns:
        Interpolated position at time t
    """
    if t1 == t0:
        return p0.copy()
    alpha = (t - t0) / (t1 - t0)
    return p0 + alpha * (p1 - p0)


def interpolate_quaternion(
    t: float,
    t0: float,
    q0: np.ndarray,
    t1: float,
    q1: np.ndarray,
) -> np.ndarray:
    """
    Spherical linear interpolation (SLERP) for quaternions.

    Args:
        t: Target timestamp
        t0: Before timestamp
        q0: Quaternion at t0 (xyzw format)
        t1: After timestamp
        q1: Quaternion at t1 (xyzw format)

    Returns:
        Interpolated quaternion at time t (xyzw format)
    """
    if t1 == t0:
        return q0.copy()

    # Create Rotation objects and use scipy's Slerp
    key_times = [t0, t1]
    key_rots = Rotation.from_quat([q0, q1])
    slerp = Slerp(key_times, key_rots)

    return slerp([t]).as_quat()[0]


def interpolate_scalar(
    t: float,
    t0: float,
    v0: float,
    t1: float,
    v1: float,
) -> float:
    """Linear interpolation of scalar value."""
    if t1 == t0:
        return v0
    alpha = (t - t0) / (t1 - t0)
    return v0 + alpha * (v1 - v0)


def find_bracketing_indices(
    timestamps: np.ndarray,
    target_time: int,
) -> Tuple[int, int]:
    """
    Find indices of samples that bracket the target time.

    Args:
        timestamps: Array of timestamps (sorted ascending)
        target_time: Target timestamp to bracket

    Returns:
        Tuple of (before_index, after_index)
    """
    idx_after = np.searchsorted(timestamps, target_time)

    # Handle edge cases
    if idx_after == 0:
        return 0, 0
    if idx_after >= len(timestamps):
        return len(timestamps) - 1, len(timestamps) - 1

    idx_before = idx_after - 1
    return idx_before, idx_after


def interpolate_trajectory_to_video(
    trajectory_table: pa.Table,
    video_timestamps: List[VideoTimestamp],
) -> List[TrajectoryFrame]:
    """
    Interpolate trajectory data to match video frame timestamps.

    Args:
        trajectory_table: Arrow table with trajectory data (50Hz)
        video_timestamps: List of video frame timestamps (60Hz)

    Returns:
        List of interpolated TrajectoryFrame objects aligned to video frames
    """
    # Convert to numpy arrays for efficient processing
    traj_timestamps = trajectory_table["timestamp_ns"].to_numpy()
    traj_positions = np.column_stack([
        trajectory_table["position_x"].to_numpy(),
        trajectory_table["position_y"].to_numpy(),
        trajectory_table["position_z"].to_numpy(),
    ])
    traj_orientations = np.column_stack([
        trajectory_table["orientation_x"].to_numpy(),
        trajectory_table["orientation_y"].to_numpy(),
        trajectory_table["orientation_z"].to_numpy(),
        trajectory_table["orientation_w"].to_numpy(),
    ])
    traj_triggers = trajectory_table["button_trigger"].to_numpy()
    traj_grips = trajectory_table["button_grip"].to_numpy()
    controller_hand = trajectory_table["controller_hand"][0].as_py()

    interpolated_frames = []

    for video_ts in video_timestamps:
        target_ns = video_ts.timestamp_ns

        # Find bracketing trajectory samples
        idx_before, idx_after = find_bracketing_indices(traj_timestamps, target_ns)

        t0 = float(traj_timestamps[idx_before])
        t1 = float(traj_timestamps[idx_after])
        t = float(target_ns)

        # Interpolate position
        pos = interpolate_position(
            t, t0, traj_positions[idx_before], t1, traj_positions[idx_after]
        )

        # Interpolate orientation (SLERP)
        quat = interpolate_quaternion(
            t, t0, traj_orientations[idx_before], t1, traj_orientations[idx_after]
        )

        # Interpolate button values
        trigger = interpolate_scalar(
            t, t0, traj_triggers[idx_before], t1, traj_triggers[idx_after]
        )
        grip = interpolate_scalar(
            t, t0, traj_grips[idx_before], t1, traj_grips[idx_after]
        )

        frame = TrajectoryFrame(
            timestamp_ns=video_ts.timestamp_ns,
            frame_index=video_ts.frame_index,
            position_x=pos[0],
            position_y=pos[1],
            position_z=pos[2],
            orientation_x=quat[0],
            orientation_y=quat[1],
            orientation_z=quat[2],
            orientation_w=quat[3],
            button_trigger=float(trigger),
            button_grip=float(grip),
            controller_hand=controller_hand,
        )
        interpolated_frames.append(frame)

    return interpolated_frames


def compute_video_trajectory_alignment(
    trajectory_timestamps: np.ndarray,
    video_timestamps: np.ndarray,
) -> List[int]:
    """
    For each video frame, find the index of the closest trajectory frame.

    Args:
        trajectory_timestamps: Array of trajectory timestamps (ns)
        video_timestamps: Array of video timestamps (ns)

    Returns:
        List of trajectory indices, one per video frame
    """
    closest_indices = []

    for video_ts in video_timestamps:
        # Find insertion point
        idx = np.searchsorted(trajectory_timestamps, video_ts)

        # Choose closest between idx-1 and idx
        if idx == 0:
            closest_idx = 0
        elif idx >= len(trajectory_timestamps):
            closest_idx = len(trajectory_timestamps) - 1
        else:
            # Compare distances
            dist_before = abs(video_ts - trajectory_timestamps[idx - 1])
            dist_after = abs(trajectory_timestamps[idx] - video_ts)
            closest_idx = idx - 1 if dist_before <= dist_after else idx

        closest_indices.append(closest_idx)

    return closest_indices


def resample_trajectory(
    trajectory_table: pa.Table,
    target_fps: float,
    source_fps: float = 50.0,
) -> pa.Table:
    """
    Resample trajectory to a different frame rate.

    Args:
        trajectory_table: Input trajectory table
        target_fps: Target frame rate
        source_fps: Source frame rate (default 50Hz)

    Returns:
        Resampled trajectory table
    """
    # Get time range
    timestamps = trajectory_table["timestamp_ns"].to_numpy()
    start_ns = timestamps[0]
    end_ns = timestamps[-1]
    duration_ns = end_ns - start_ns

    # Generate new timestamps at target FPS
    num_frames = int((duration_ns / 1e9) * target_fps) + 1
    new_timestamps_ns = np.linspace(start_ns, end_ns, num_frames, dtype=np.int64)

    # Create video timestamps for interpolation
    video_timestamps = [
        VideoTimestamp(frame_index=i, timestamp_ns=int(ts), closest_trajectory_idx=0)
        for i, ts in enumerate(new_timestamps_ns)
    ]

    # Interpolate
    interpolated = interpolate_trajectory_to_video(trajectory_table, video_timestamps)

    # Convert back to table
    from .schemas import trajectory_frames_to_table
    return trajectory_frames_to_table(interpolated)


def estimate_trajectory_fps(trajectory_table: pa.Table) -> float:
    """
    Estimate the frame rate of trajectory data.

    Args:
        trajectory_table: Trajectory data table

    Returns:
        Estimated FPS
    """
    timestamps = trajectory_table["timestamp_ns"].to_numpy()
    if len(timestamps) < 2:
        return 0.0

    # Calculate median interval
    intervals_ns = np.diff(timestamps)
    median_interval_ns = np.median(intervals_ns)

    if median_interval_ns == 0:
        return 0.0

    return 1e9 / median_interval_ns


def validate_sync_quality(
    trajectory_timestamps: np.ndarray,
    video_timestamps: np.ndarray,
    max_drift_ms: float = 50.0,
) -> Tuple[bool, float, str]:
    """
    Validate synchronization quality between trajectory and video.

    Args:
        trajectory_timestamps: Trajectory timestamps (ns)
        video_timestamps: Video timestamps (ns)
        max_drift_ms: Maximum acceptable drift in milliseconds

    Returns:
        Tuple of (is_valid, max_drift_ms, message)
    """
    if len(trajectory_timestamps) == 0 or len(video_timestamps) == 0:
        return False, float("inf"), "Empty timestamp arrays"

    # Check overlap
    traj_start, traj_end = trajectory_timestamps[0], trajectory_timestamps[-1]
    video_start, video_end = video_timestamps[0], video_timestamps[-1]

    overlap_start = max(traj_start, video_start)
    overlap_end = min(traj_end, video_end)

    if overlap_end <= overlap_start:
        return False, float("inf"), "No temporal overlap between trajectory and video"

    # Calculate drift at boundaries
    start_drift_ms = abs(traj_start - video_start) / 1e6
    end_drift_ms = abs(traj_end - video_end) / 1e6
    actual_max_drift = max(start_drift_ms, end_drift_ms)

    if actual_max_drift > max_drift_ms:
        return (
            False,
            actual_max_drift,
            f"Drift exceeds threshold: {actual_max_drift:.1f}ms > {max_drift_ms}ms",
        )

    return True, actual_max_drift, f"Sync OK, max drift: {actual_max_drift:.1f}ms"
