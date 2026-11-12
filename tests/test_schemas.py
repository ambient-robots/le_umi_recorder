"""
Tests for umi.pipeline.schemas module.

Tests data model validation, Arrow table conversions, and configuration handling.
"""

import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from umi.pipeline.schemas import (
    TrajectoryFrame,
    VideoTimestamp,
    JointState,
    GripperState,
    TaskPrompt,
    StageMetadata,
    PipelineConfig,
    Stage1Config,
    Stage2Config,
    Stage3Config,
    Stage4Config,
    AssemblyConfig,
    CameraConfig,
    trajectory_frames_to_table,
    video_timestamps_to_table,
    joint_states_to_table,
    gripper_states_to_table,
    TRAJECTORY_SCHEMA,
    VIDEO_TIMESTAMPS_SCHEMA,
    JOINTS_SCHEMA,
    GRIPPER_STATES_SCHEMA,
)


class TestTrajectoryFrame:
    """Tests for TrajectoryFrame dataclass."""

    def test_to_dict_all_fields(self):
        """Test that to_dict includes all fields."""
        frame = TrajectoryFrame(
            timestamp_ns=1000000000,
            frame_index=42,
            position_x=0.1,
            position_y=0.2,
            position_z=0.3,
            orientation_x=0.0,
            orientation_y=0.0,
            orientation_z=0.0,
            orientation_w=1.0,
            button_trigger=0.5,
            button_grip=0.8,
            controller_hand="right",
        )

        d = frame.to_dict()

        assert d["timestamp_ns"] == 1000000000
        assert d["frame_index"] == 42
        assert d["position_x"] == 0.1
        assert d["position_y"] == 0.2
        assert d["position_z"] == 0.3
        assert d["orientation_x"] == 0.0
        assert d["orientation_y"] == 0.0
        assert d["orientation_z"] == 0.0
        assert d["orientation_w"] == 1.0
        assert d["button_trigger"] == 0.5
        assert d["button_grip"] == 0.8
        assert d["controller_hand"] == "right"

    def test_to_dict_preserves_types(self):
        """Test that to_dict preserves correct types."""
        frame = TrajectoryFrame(
            timestamp_ns=1000000000,
            frame_index=0,
            position_x=0.0,
            position_y=0.0,
            position_z=0.0,
            orientation_x=0.0,
            orientation_y=0.0,
            orientation_z=0.0,
            orientation_w=1.0,
            button_trigger=0.0,
            button_grip=0.0,
            controller_hand="left",
        )

        d = frame.to_dict()

        assert isinstance(d["timestamp_ns"], int)
        assert isinstance(d["frame_index"], int)
        assert isinstance(d["position_x"], float)
        assert isinstance(d["controller_hand"], str)


class TestVideoTimestamp:
    """Tests for VideoTimestamp dataclass."""

    def test_to_dict(self):
        """Test VideoTimestamp to_dict."""
        ts = VideoTimestamp(frame_index=10, timestamp_ns=2000000000, closest_trajectory_idx=8)

        d = ts.to_dict()

        assert d["frame_index"] == 10
        assert d["timestamp_ns"] == 2000000000
        assert d["closest_trajectory_idx"] == 8


class TestJointState:
    """Tests for JointState dataclass."""

    def test_to_dict(self):
        """Test JointState to_dict includes all fields."""
        state = JointState(
            timestamp_ns=1000000000,
            frame_index=5,
            shoulder_pan=0.1,
            shoulder_lift=-0.2,
            elbow_flex=0.3,
            elbow_roll=-0.4,
            wrist_flex=0.5,
            wrist_roll=-0.6,
            ik_success=True,
            ik_error=0.001,
        )

        d = state.to_dict()

        assert d["shoulder_pan"] == 0.1
        assert d["shoulder_lift"] == -0.2
        assert d["elbow_flex"] == 0.3
        assert d["elbow_roll"] == -0.4
        assert d["wrist_flex"] == 0.5
        assert d["wrist_roll"] == -0.6
        assert d["ik_success"] is True
        assert d["ik_error"] == 0.001

    def test_to_array_order(self):
        """Test JointState to_array returns correct order."""
        state = JointState(
            timestamp_ns=1000000000,
            frame_index=0,
            shoulder_pan=0.1,
            shoulder_lift=0.2,
            elbow_flex=0.3,
            elbow_roll=0.4,
            wrist_flex=0.5,
            wrist_roll=0.6,
            ik_success=True,
            ik_error=0.0,
        )

        arr = state.to_array()

        assert len(arr) == 6
        assert arr == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    def test_to_array_no_gripper(self):
        """Test that to_array does not include gripper (6 joints only)."""
        state = JointState(
            timestamp_ns=0,
            frame_index=0,
            shoulder_pan=0.0,
            shoulder_lift=0.0,
            elbow_flex=0.0,
            elbow_roll=0.0,
            wrist_flex=0.0,
            wrist_roll=0.0,
            ik_success=True,
            ik_error=0.0,
        )

        arr = state.to_array()

        assert len(arr) == 6  # No gripper


class TestGripperState:
    """Tests for GripperState dataclass."""

    def test_to_dict(self):
        """Test GripperState to_dict."""
        state = GripperState(
            frame_index=100,
            timestamp_ns=5000000000,
            measured_state=30.0,
            commanded_state=0.0,
            confidence=0.95,
            annotation="grasping",
        )

        d = state.to_dict()

        assert d["frame_index"] == 100
        assert d["measured_state"] == 30.0
        assert d["commanded_state"] == 0.0
        assert d["confidence"] == 0.95
        assert d["annotation"] == "grasping"

    def test_default_annotation(self):
        """Test GripperState default annotation is empty string."""
        state = GripperState(
            frame_index=0,
            timestamp_ns=0,
            measured_state=50.0,
            commanded_state=50.0,
            confidence=0.5,
        )

        assert state.annotation == ""


class TestTaskPrompt:
    """Tests for TaskPrompt dataclass."""

    def test_to_dict(self):
        """Test TaskPrompt to_dict."""
        prompt = TaskPrompt(
            task_description="Pick up the cube",
            confidence=0.92,
            model="gemini-2.0-flash",
            generation_timestamp="2025-01-01T00:00:00",
            video_summary="Robot grasping a cube",
            objects=["cube", "table"],
            actions=["pick", "lift"],
        )

        d = prompt.to_dict()

        assert d["task_description"] == "Pick up the cube"
        assert d["confidence"] == 0.92
        assert d["model"] == "gemini-2.0-flash"
        assert d["objects"] == ["cube", "table"]
        assert d["actions"] == ["pick", "lift"]

    def test_default_lists(self):
        """Test TaskPrompt default lists are empty."""
        prompt = TaskPrompt(
            task_description="Test",
            confidence=0.5,
            model="test",
            generation_timestamp="2025-01-01",
        )

        assert prompt.objects == []
        assert prompt.actions == []


class TestStageMetadata:
    """Tests for StageMetadata dataclass."""

    def test_auto_timestamp(self):
        """Test that created_at is auto-generated."""
        metadata = StageMetadata(stage=1, version="1.0.0")

        assert metadata.created_at != ""
        assert "T" in metadata.created_at  # ISO format

    def test_to_dict_includes_extra(self):
        """Test that to_dict includes extra fields."""
        metadata = StageMetadata(
            stage=2,
            version="1.0.0",
            extra={"custom_field": "value", "frames_processed": 100},
        )

        d = metadata.to_dict()

        assert d["stage"] == 2
        assert d["version"] == "1.0.0"
        assert d["custom_field"] == "value"
        assert d["frames_processed"] == 100


class TestArrowTableConversions:
    """Tests for Arrow table conversion functions."""

    def test_trajectory_frames_to_table(self, sample_trajectory_frames):
        """Test converting trajectory frames to Arrow table."""
        table = trajectory_frames_to_table(sample_trajectory_frames)

        assert len(table) == len(sample_trajectory_frames)
        assert table.schema.equals(TRAJECTORY_SCHEMA)

    def test_trajectory_frames_to_table_empty(self):
        """Test converting empty list to Arrow table."""
        table = trajectory_frames_to_table([])

        assert len(table) == 0
        assert table.schema.equals(TRAJECTORY_SCHEMA)

    def test_video_timestamps_to_table(self, sample_video_timestamps):
        """Test converting video timestamps to Arrow table."""
        table = video_timestamps_to_table(sample_video_timestamps)

        assert len(table) == len(sample_video_timestamps)
        assert table.schema.equals(VIDEO_TIMESTAMPS_SCHEMA)

    def test_joint_states_to_table(self, sample_joint_states):
        """Test converting joint states to Arrow table."""
        table = joint_states_to_table(sample_joint_states)

        assert len(table) == len(sample_joint_states)
        assert table.schema.equals(JOINTS_SCHEMA)

    def test_gripper_states_to_table(self, sample_gripper_states):
        """Test converting gripper states to Arrow table."""
        table = gripper_states_to_table(sample_gripper_states)

        assert len(table) == len(sample_gripper_states)
        assert table.schema.equals(GRIPPER_STATES_SCHEMA)


class TestArrowSchemas:
    """Tests for Arrow schema definitions."""

    def test_trajectory_schema_fields(self):
        """Test TRAJECTORY_SCHEMA has correct fields."""
        field_names = [f.name for f in TRAJECTORY_SCHEMA]

        assert "timestamp_ns" in field_names
        assert "frame_index" in field_names
        assert "position_x" in field_names
        assert "position_y" in field_names
        assert "position_z" in field_names
        assert "orientation_x" in field_names
        assert "orientation_y" in field_names
        assert "orientation_z" in field_names
        assert "orientation_w" in field_names
        assert "button_trigger" in field_names
        assert "button_grip" in field_names
        assert "controller_hand" in field_names

    def test_trajectory_schema_types(self):
        """Test TRAJECTORY_SCHEMA has correct types."""
        assert TRAJECTORY_SCHEMA.field("timestamp_ns").type == pa.int64()
        assert TRAJECTORY_SCHEMA.field("position_x").type == pa.float64()
        assert TRAJECTORY_SCHEMA.field("button_trigger").type == pa.float32()
        assert TRAJECTORY_SCHEMA.field("controller_hand").type == pa.string()

    def test_joints_schema_fields(self):
        """Test JOINTS_SCHEMA has correct fields."""
        field_names = [f.name for f in JOINTS_SCHEMA]

        assert "shoulder_pan" in field_names
        assert "shoulder_lift" in field_names
        assert "elbow_flex" in field_names
        assert "elbow_roll" in field_names
        assert "wrist_flex" in field_names
        assert "wrist_roll" in field_names
        assert "ik_success" in field_names
        assert "ik_error" in field_names

    def test_gripper_schema_fields(self):
        """Test GRIPPER_STATES_SCHEMA has correct fields."""
        field_names = [f.name for f in GRIPPER_STATES_SCHEMA]

        assert "measured_state" in field_names
        assert "commanded_state" in field_names
        assert "confidence" in field_names
        assert "annotation" in field_names


class TestPipelineConfig:
    """Tests for PipelineConfig loading and saving."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.stage1.trajectory_fps == 50
        assert config.stage1.video_fps == 60
        assert config.stage1.controller_hand == "right"
        assert config.stage2.solver_dt == pytest.approx(0.0167, abs=0.001)
        assert config.stage3.model == "gemini-2.0-flash"
        assert config.stage4.model == "gemini-2.0-flash"
        assert config.assembly.fps == 60

    def test_yaml_roundtrip(self, sample_pipeline_config):
        """Test saving and loading config from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"

            # Modify some values
            sample_pipeline_config.stage1.trajectory_fps = 100
            sample_pipeline_config.stage3.model = "gemini-pro"

            # Save
            sample_pipeline_config.to_yaml(yaml_path)

            # Load
            loaded = PipelineConfig.from_yaml(yaml_path)

            # Verify
            assert loaded.stage1.trajectory_fps == 100
            assert loaded.stage3.model == "gemini-pro"

    def test_from_yaml_partial(self):
        """Test loading config with partial YAML."""
        import yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "partial_config.yaml"

            # Write partial config
            partial = {
                "stage1": {"video_fps": 30},
                "assembly": {"repo_id": "user/my_dataset"},
            }
            with open(yaml_path, "w") as f:
                yaml.safe_dump(partial, f)

            # Load
            config = PipelineConfig.from_yaml(yaml_path)

            # Verify overridden values
            assert config.stage1.video_fps == 30
            assert config.assembly.repo_id == "user/my_dataset"

            # Verify defaults
            assert config.stage1.trajectory_fps == 50
            assert config.stage2.solver_dt == pytest.approx(0.0167, abs=0.001)


class TestCameraConfig:
    """Tests for CameraConfig dataclass."""

    def test_default_values(self):
        """Test CameraConfig default values."""
        cam = CameraConfig(name="wrist", index=0)

        assert cam.width == 640
        assert cam.height == 480
        assert cam.fps == 60

    def test_custom_values(self):
        """Test CameraConfig custom values."""
        cam = CameraConfig(
            name="overhead",
            index=1,
            width=1280,
            height=720,
            fps=30,
        )

        assert cam.name == "overhead"
        assert cam.index == 1
        assert cam.width == 1280
        assert cam.height == 720
        assert cam.fps == 30


class TestStageConfigs:
    """Tests for individual stage configuration dataclasses."""

    def test_stage1_config_defaults(self):
        """Test Stage1Config defaults."""
        config = Stage1Config()

        assert config.trajectory_fps == 50
        assert config.video_fps == 60
        assert config.controller_hand == "right"
        assert config.server_url == "https://localhost:8000"
        assert config.camera.name == "wrist"

    def test_stage2_config_defaults(self):
        """Test Stage2Config defaults."""
        config = Stage2Config()

        assert config.solver_dt == pytest.approx(0.0167, abs=0.001)
        assert config.regularization == 1e-4
        assert config.frame_name == "gripper_frame"
        assert len(config.joint_names) == 6

    def test_stage3_config_defaults(self):
        """Test Stage3Config defaults."""
        config = Stage3Config()

        assert config.model == "gemini-2.0-flash"
        assert config.prompt_template == "task_v1"
        assert config.api_key_env == "GEMINI_API_KEY"

    def test_stage4_config_defaults(self):
        """Test Stage4Config defaults."""
        config = Stage4Config()

        assert config.model == "gemini-2.0-flash"
        assert config.batch_size == 60
        assert config.sample_rate == 1

    def test_assembly_config_defaults(self):
        """Test AssemblyConfig defaults."""
        config = AssemblyConfig()

        assert config.output_format == "lerobot_v2.1"
        assert config.include_raw_poses is False
        assert config.fps == 60
