"""
Shared pytest fixtures for UMI pipeline tests.
"""

import json
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Import pipeline schemas
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock modules that may not be installed before importing pipeline
sys.modules.setdefault('rerun', MagicMock())
sys.modules.setdefault('placo', MagicMock())
sys.modules.setdefault('google.generativeai', MagicMock())
sys.modules.setdefault('google.generativeai.types', MagicMock())

from umi.pipeline.schemas import (
    TrajectoryFrame,
    VideoTimestamp,
    JointState,
    GripperState,
    TaskPrompt,
    StageMetadata,
    PipelineConfig,
    trajectory_frames_to_table,
    video_timestamps_to_table,
    joint_states_to_table,
    gripper_states_to_table,
    TRAJECTORY_SCHEMA,
    VIDEO_TIMESTAMPS_SCHEMA,
    JOINTS_SCHEMA,
    GRIPPER_STATES_SCHEMA,
)


# =============================================================================
# Trajectory Fixtures
# =============================================================================

@pytest.fixture
def sample_trajectory_frames() -> List[TrajectoryFrame]:
    """Generate 100 sample trajectory frames at ~50Hz (20ms intervals)."""
    base_time_ns = 1000000000000  # 1 second in nanoseconds
    interval_ns = 20_000_000  # 20ms = 50Hz

    frames = []
    for i in range(100):
        # Create a simple circular motion for testing
        t = i / 50.0  # seconds
        angle = t * 2 * np.pi / 4  # One rotation per 4 seconds

        frame = TrajectoryFrame(
            timestamp_ns=base_time_ns + i * interval_ns,
            frame_index=i,
            position_x=0.3 + 0.1 * np.cos(angle),
            position_y=0.1 * np.sin(angle),
            position_z=0.2,
            orientation_x=0.0,
            orientation_y=0.0,
            orientation_z=np.sin(angle / 2),
            orientation_w=np.cos(angle / 2),
            button_trigger=0.5 if i % 20 < 10 else 0.0,
            button_grip=0.8 if i > 50 else 0.0,
            controller_hand="right",
        )
        frames.append(frame)

    return frames


@pytest.fixture
def sample_trajectory_table(sample_trajectory_frames) -> pa.Table:
    """Convert sample frames to Arrow table."""
    return trajectory_frames_to_table(sample_trajectory_frames)


# =============================================================================
# Video Timestamp Fixtures
# =============================================================================

@pytest.fixture
def sample_video_timestamps() -> List[VideoTimestamp]:
    """Generate 120 video timestamps at ~60Hz (16.67ms intervals)."""
    base_time_ns = 1000000000000  # Match trajectory start
    interval_ns = 16_666_667  # ~60Hz

    timestamps = []
    for i in range(120):
        # Calculate closest trajectory frame (50Hz vs 60Hz)
        video_time = i * interval_ns
        traj_interval = 20_000_000  # 50Hz
        closest_traj = int(round(video_time / traj_interval))
        closest_traj = min(closest_traj, 99)  # Cap at max trajectory index

        ts = VideoTimestamp(
            frame_index=i,
            timestamp_ns=base_time_ns + i * interval_ns,
            closest_trajectory_idx=closest_traj,
        )
        timestamps.append(ts)

    return timestamps


@pytest.fixture
def sample_video_timestamps_table(sample_video_timestamps) -> pa.Table:
    """Convert sample video timestamps to Arrow table."""
    return video_timestamps_to_table(sample_video_timestamps)


# =============================================================================
# Joint State Fixtures
# =============================================================================

@pytest.fixture
def sample_joint_states() -> List[JointState]:
    """Generate sample joint states aligned to video frames."""
    base_time_ns = 1000000000000
    interval_ns = 16_666_667  # 60Hz

    states = []
    for i in range(120):
        t = i / 60.0

        state = JointState(
            timestamp_ns=base_time_ns + i * interval_ns,
            frame_index=i,
            shoulder_pan=0.5 * np.sin(t),
            shoulder_lift=-0.3,
            elbow_flex=0.8 + 0.2 * np.sin(t * 2),
            elbow_roll=0.0,
            wrist_flex=0.1 * np.cos(t),
            wrist_roll=0.0,
            ik_success=True,
            ik_error=0.001,
        )
        states.append(state)

    return states


@pytest.fixture
def sample_joints_table(sample_joint_states) -> pa.Table:
    """Convert sample joint states to Arrow table."""
    return joint_states_to_table(sample_joint_states)


# =============================================================================
# Gripper State Fixtures
# =============================================================================

@pytest.fixture
def sample_gripper_states() -> List[GripperState]:
    """Generate sample gripper states with varying open/close patterns."""
    base_time_ns = 1000000000000
    interval_ns = 16_666_667  # 60Hz

    states = []
    for i in range(120):
        # Simulate a grasp action: open -> close -> hold -> open
        if i < 30:
            measured = 100.0  # Fully open
            commanded = 100.0
            annotation = "open"
        elif i < 50:
            progress = (i - 30) / 20.0
            measured = 100.0 - 70.0 * progress  # Closing
            commanded = 0.0
            annotation = "closing"
        elif i < 90:
            measured = 30.0  # Holding object (can't fully close)
            commanded = 0.0
            annotation = "grasping"
        else:
            progress = (i - 90) / 30.0
            measured = 30.0 + 70.0 * progress
            commanded = 100.0
            annotation = "releasing"

        state = GripperState(
            frame_index=i,
            timestamp_ns=base_time_ns + i * interval_ns,
            measured_state=measured,
            commanded_state=commanded,
            confidence=0.9,
            annotation=annotation,
        )
        states.append(state)

    return states


@pytest.fixture
def sample_gripper_table(sample_gripper_states) -> pa.Table:
    """Convert sample gripper states to Arrow table."""
    return gripper_states_to_table(sample_gripper_states)


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def sample_pipeline_config() -> PipelineConfig:
    """Create a sample pipeline configuration."""
    return PipelineConfig()


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_episode_dir(
    sample_trajectory_table,
    sample_video_timestamps_table,
    sample_joints_table,
    sample_gripper_states,
):
    """Create a temporary episode directory with stage outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        episode_dir = Path(tmpdir) / "episode_000"

        # Stage 1 outputs
        stage1_dir = episode_dir / "stage1"
        stage1_dir.mkdir(parents=True)
        pq.write_table(sample_trajectory_table, stage1_dir / "raw_trajectory.parquet")
        pq.write_table(sample_video_timestamps_table, stage1_dir / "video_timestamps.parquet")

        # Create a dummy video file (just metadata, no actual video)
        (stage1_dir / "raw_video.mp4").write_bytes(b"dummy_video_data")

        # Stage 2 outputs
        stage2_dir = episode_dir / "stage2"
        stage2_dir.mkdir(parents=True)
        pq.write_table(sample_joints_table, stage2_dir / "joints.parquet")

        # Stage 3 outputs
        stage3_dir = episode_dir / "stage3"
        stage3_dir.mkdir(parents=True)
        task_prompt = TaskPrompt(
            task_description="Pick up the red block and place it in the blue box",
            confidence=0.95,
            model="gemini-2.0-flash",
            generation_timestamp="2025-01-01T00:00:00",
            video_summary="Robot arm grasping a red block",
            objects=["red block", "blue box"],
            actions=["pick", "place"],
        )
        with open(stage3_dir / "task_prompt.json", "w") as f:
            json.dump(task_prompt.to_dict(), f)

        # Stage 4 outputs
        stage4_dir = episode_dir / "stage4"
        stage4_dir.mkdir(parents=True)
        gripper_table = gripper_states_to_table(sample_gripper_states)
        pq.write_table(gripper_table, stage4_dir / "gripper_states.parquet")

        yield episode_dir


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Gemini API Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_gemini_task_response():
    """Mock Gemini API response for task prompt generation."""
    return {
        "task_description": "Pick up the red cube and place it on the blue platform",
        "confidence": 0.92,
        "video_summary": "The robot arm moves to grasp a red cube from the table and transfers it to a raised platform",
        "objects": ["red cube", "blue platform", "table"],
        "actions": ["approach", "grasp", "lift", "move", "place", "release"],
    }


@pytest.fixture
def mock_gemini_gripper_response():
    """Mock Gemini API response for gripper detection."""
    return {
        "total_duration_seconds": 2.0,
        "summary": "Gripper opens, closes to grasp object, holds, then releases",
        "frames": [
            {"time_seconds": 0.0, "measured": 100, "commanded": 100, "annotation": "open"},
            {"time_seconds": 0.5, "measured": 100, "commanded": 0, "annotation": "closing"},
            {"time_seconds": 0.8, "measured": 30, "commanded": 0, "annotation": "grasping"},
            {"time_seconds": 1.5, "measured": 30, "commanded": 0, "annotation": "holding"},
            {"time_seconds": 1.8, "measured": 80, "commanded": 100, "annotation": "releasing"},
            {"time_seconds": 2.0, "measured": 100, "commanded": 100, "annotation": "open"},
        ],
    }


# =============================================================================
# Numpy Array Fixtures for Interpolation Tests
# =============================================================================

@pytest.fixture
def trajectory_timestamps_array(sample_trajectory_frames) -> np.ndarray:
    """Extract timestamps as numpy array."""
    return np.array([f.timestamp_ns for f in sample_trajectory_frames])


@pytest.fixture
def video_timestamps_array(sample_video_timestamps) -> np.ndarray:
    """Extract video timestamps as numpy array."""
    return np.array([t.timestamp_ns for t in sample_video_timestamps])
