"""
Tests for umi.pipeline.stage2_ik_solver module.

Uses mocking for placo and rerun dependencies.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from umi.pipeline.schemas import (
    Stage2Config,
    TrajectoryFrame,
    VideoTimestamp,
    trajectory_frames_to_table,
    video_timestamps_to_table,
)


class TestIKPostProcessorInit:
    """Tests for IKPostProcessor initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        # Mock placo and rerun to avoid import errors
        with patch.dict('sys.modules', {'placo': MagicMock(), 'rerun': MagicMock()}):
            from umi.pipeline.stage2_ik_solver import IKPostProcessor

            config = Stage2Config(urdf_path="/path/to/robot.urdf")
            processor = IKPostProcessor(config)

            assert processor.urdf_path == "/path/to/robot.urdf"
            assert len(processor.arm_joint_names) == 6
            assert processor.ik_robot is None  # Not set until setup()

    def test_init_custom_joint_names(self):
        """Test initialization with custom joint names."""
        with patch.dict('sys.modules', {'placo': MagicMock(), 'rerun': MagicMock()}):
            from umi.pipeline.stage2_ik_solver import IKPostProcessor

            custom_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
            config = Stage2Config(urdf_path="/path/to/robot.urdf", joint_names=custom_joints)
            processor = IKPostProcessor(config)

            assert processor.arm_joint_names == custom_joints


class TestCoordinateTransforms:
    """Tests for coordinate frame transformations."""

    def test_setup_transforms_creates_matrices(self):
        """Test that _setup_transforms creates valid transformation matrices."""
        with patch.dict('sys.modules', {'placo': MagicMock(), 'rerun': MagicMock()}):
            from umi.pipeline.stage2_ik_solver import IKPostProcessor

            config = Stage2Config(urdf_path="/path/to/robot.urdf")
            processor = IKPostProcessor(config)

            # Check transformation matrices are 4x4
            assert processor.T_tracker_to_normalized.shape == (4, 4)
            assert processor.T_normalized_to_tool.shape == (4, 4)

            # Check robot base rotation is 3x3
            assert processor.robot_base_rotation.shape == (3, 3)

            # Check tool-to-gripper rotation is 3x3
            assert processor.tool_to_gripper_rotation.shape == (3, 3)

    def test_tracker_to_normalized_is_rotation(self):
        """Test that T_tracker_to_normalized is a valid rotation matrix."""
        with patch.dict('sys.modules', {'placo': MagicMock(), 'rerun': MagicMock()}):
            from umi.pipeline.stage2_ik_solver import IKPostProcessor

            config = Stage2Config(urdf_path="/path/to/robot.urdf")
            processor = IKPostProcessor(config)

            # Extract rotation part
            R = processor.T_tracker_to_normalized[:3, :3]

            # Check orthogonality (R @ R.T = I)
            np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=10)

            # Check determinant = 1 (proper rotation)
            assert np.linalg.det(R) == pytest.approx(1.0)

    def test_robot_base_rotation_180_degrees(self):
        """Test that robot base is rotated 180° around Z."""
        with patch.dict('sys.modules', {'placo': MagicMock(), 'rerun': MagicMock()}):
            from umi.pipeline.stage2_ik_solver import IKPostProcessor

            config = Stage2Config(urdf_path="/path/to/robot.urdf")
            processor = IKPostProcessor(config)

            # Apply rotation to X axis [1, 0, 0]
            rotated = processor.robot_base_rotation @ np.array([1, 0, 0])

            # Should point in -X direction
            np.testing.assert_array_almost_equal(rotated, [-1, 0, 0], decimal=10)


class TestRelativeRotation:
    """Tests for relative rotation computation."""

    def test_compute_relative_rotvec_identity(self):
        """Test that identical quaternions give zero rotation."""
        with patch.dict('sys.modules', {'placo': MagicMock(), 'rerun': MagicMock()}):
            from umi.pipeline.stage2_ik_solver import IKPostProcessor

            config = Stage2Config(urdf_path="/path/to/robot.urdf")
            processor = IKPostProcessor(config)

            q = np.array([0, 0, 0, 1])  # Identity
            rotvec = processor._compute_relative_rotvec(q, q)

            np.testing.assert_array_almost_equal(rotvec, [0, 0, 0], decimal=10)

    def test_compute_relative_rotvec_90_degrees_z(self):
        """Test 90° rotation around Z axis."""
        with patch.dict('sys.modules', {'placo': MagicMock(), 'rerun': MagicMock()}):
            from umi.pipeline.stage2_ik_solver import IKPostProcessor

            config = Stage2Config(urdf_path="/path/to/robot.urdf")
            processor = IKPostProcessor(config)

            q0 = np.array([0, 0, 0, 1])  # Identity
            q1 = Rotation.from_euler('z', 90, degrees=True).as_quat()

            rotvec = processor._compute_relative_rotvec(q0, q1)

            # Should be approximately [0, 0, π/2]
            expected = np.array([0, 0, np.pi / 2])
            np.testing.assert_array_almost_equal(rotvec, expected, decimal=5)

    def test_compute_relative_rotvec_handles_double_cover(self):
        """Test that quaternion double cover is handled correctly."""
        with patch.dict('sys.modules', {'placo': MagicMock(), 'rerun': MagicMock()}):
            from umi.pipeline.stage2_ik_solver import IKPostProcessor

            config = Stage2Config(urdf_path="/path/to/robot.urdf")
            processor = IKPostProcessor(config)

            q0 = np.array([0, 0, 0, 1])  # Identity
            q1 = np.array([0, 0, 0, -1])  # Same rotation, opposite quaternion

            rotvec = processor._compute_relative_rotvec(q0, q1)

            # Should be close to zero (same rotation)
            assert np.linalg.norm(rotvec) < 0.01


class TestIKSolverSetup:
    """Tests for IK solver setup with mocked placo."""

    def test_setup_initializes_solver(self):
        """Test that setup initializes the IK solver."""
        mock_placo = MagicMock()
        mock_robot = MagicMock()
        mock_solver = MagicMock()
        mock_task = MagicMock()

        mock_placo.RobotWrapper.return_value = mock_robot
        mock_placo.KinematicsSolver.return_value = mock_solver
        mock_solver.add_frame_task.return_value = mock_task
        mock_robot.get_T_world_frame.return_value = np.eye(4)

        with patch.dict('sys.modules', {'placo': mock_placo, 'rerun': MagicMock()}):
            from umi.pipeline.stage2_ik_solver import IKPostProcessor

            config = Stage2Config(urdf_path="/path/to/robot.urdf")
            processor = IKPostProcessor(config)
            processor.setup()

            assert processor.ik_robot is not None
            assert processor.ik_solver is not None
            assert processor.ik_task is not None
            mock_placo.RobotWrapper.assert_called_once()

    def test_setup_without_urdf_raises(self):
        """Test that setup without URDF path raises error."""
        with patch.dict('sys.modules', {'placo': MagicMock(), 'rerun': MagicMock()}):
            from umi.pipeline.stage2_ik_solver import IKPostProcessor

            config = Stage2Config(urdf_path="")  # Empty path
            processor = IKPostProcessor(config)

            with pytest.raises(ValueError) as exc_info:
                processor.setup()

            assert "URDF path is required" in str(exc_info.value)


class TestIKSolving:
    """Tests for IK solving with mocked placo."""

    @pytest.fixture
    def mock_processor(self):
        """Create an IK processor with mocked dependencies."""
        mock_placo = MagicMock()
        mock_robot = MagicMock()
        mock_solver = MagicMock()
        mock_task = MagicMock()

        mock_placo.RobotWrapper.return_value = mock_robot
        mock_placo.KinematicsSolver.return_value = mock_solver
        mock_solver.add_frame_task.return_value = mock_task

        # Mock joint positions
        mock_robot.get_joint.side_effect = lambda name: 0.0
        mock_robot.get_T_world_frame.return_value = np.eye(4)

        with patch.dict('sys.modules', {'placo': mock_placo, 'rerun': MagicMock()}):
            from umi.pipeline.stage2_ik_solver import IKPostProcessor

            config = Stage2Config(urdf_path="/path/to/robot.urdf")
            processor = IKPostProcessor(config)
            processor.setup()

            yield processor

    def test_solve_ik_without_setup_raises(self):
        """Test that solving IK without setup raises error."""
        with patch.dict('sys.modules', {'placo': MagicMock(), 'rerun': MagicMock()}):
            from umi.pipeline.stage2_ik_solver import IKPostProcessor

            config = Stage2Config(urdf_path="/path/to/robot.urdf")
            processor = IKPostProcessor(config)

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
                controller_hand="right",
            )

            with pytest.raises(RuntimeError) as exc_info:
                processor.solve_ik(frame)

            assert "not initialized" in str(exc_info.value)

    def test_solve_ik_returns_joint_state(self, mock_processor):
        """Test that solve_ik returns valid JointState."""
        frame = TrajectoryFrame(
            timestamp_ns=1000000000,
            frame_index=0,
            position_x=0.3,
            position_y=0.0,
            position_z=0.2,
            orientation_x=0.0,
            orientation_y=0.0,
            orientation_z=0.0,
            orientation_w=1.0,
            button_trigger=0.0,
            button_grip=0.0,
            controller_hand="right",
        )

        result = mock_processor.solve_ik(frame)

        # Check result has correct attributes
        assert result.timestamp_ns == 1000000000
        assert result.frame_index == 0
        assert hasattr(result, 'shoulder_pan')
        assert hasattr(result, 'ik_success')
        assert hasattr(result, 'ik_error')

    def test_solve_ik_first_frame_sets_origin(self, mock_processor):
        """Test that first frame sets the origin."""
        frame = TrajectoryFrame(
            timestamp_ns=1000000000,
            frame_index=0,
            position_x=0.5,
            position_y=0.1,
            position_z=0.3,
            orientation_x=0.0,
            orientation_y=0.0,
            orientation_z=0.0,
            orientation_w=1.0,
            button_trigger=0.0,
            button_grip=0.0,
            controller_hand="right",
        )

        assert mock_processor.origin_position is None

        mock_processor.solve_ik(frame)

        assert mock_processor.origin_position is not None

    def test_reset_clears_origin(self, mock_processor):
        """Test that reset clears the origin."""
        frame = TrajectoryFrame(
            timestamp_ns=1000000000,
            frame_index=0,
            position_x=0.5,
            position_y=0.1,
            position_z=0.3,
            orientation_x=0.0,
            orientation_y=0.0,
            orientation_z=0.0,
            orientation_w=1.0,
            button_trigger=0.0,
            button_grip=0.0,
            controller_hand="right",
        )

        mock_processor.solve_ik(frame)
        assert mock_processor.origin_position is not None

        mock_processor.reset()
        assert mock_processor.origin_position is None


class TestProcessEpisode:
    """Tests for processing complete episodes."""

    def test_process_episode_missing_trajectory_raises(self):
        """Test that missing trajectory file raises error."""
        with patch.dict('sys.modules', {'placo': MagicMock(), 'rerun': MagicMock()}):
            from umi.pipeline.stage2_ik_solver import IKPostProcessor

            config = Stage2Config(urdf_path="/path/to/robot.urdf")
            processor = IKPostProcessor(config)

            with tempfile.TemporaryDirectory() as tmpdir:
                stage1_dir = Path(tmpdir) / "stage1"
                stage1_dir.mkdir()
                output_dir = Path(tmpdir) / "stage2"

                with pytest.raises(FileNotFoundError) as exc_info:
                    processor.process_episode(stage1_dir, output_dir)

                assert "Trajectory file not found" in str(exc_info.value)

    def test_process_episode_creates_output_files(self, sample_trajectory_table, sample_video_timestamps_table):
        """Test that process_episode creates expected output files."""
        mock_placo = MagicMock()
        mock_robot = MagicMock()
        mock_solver = MagicMock()
        mock_task = MagicMock()

        mock_placo.RobotWrapper.return_value = mock_robot
        mock_placo.KinematicsSolver.return_value = mock_solver
        mock_solver.add_frame_task.return_value = mock_task
        mock_robot.get_joint.side_effect = lambda name: 0.0
        mock_robot.get_T_world_frame.return_value = np.eye(4)

        mock_rr = MagicMock()

        with patch.dict('sys.modules', {'placo': mock_placo, 'rerun': mock_rr}):
            from umi.pipeline.stage2_ik_solver import IKPostProcessor

            with tempfile.TemporaryDirectory() as tmpdir:
                # Create stage1 inputs
                stage1_dir = Path(tmpdir) / "stage1"
                stage1_dir.mkdir()
                pq.write_table(sample_trajectory_table, stage1_dir / "raw_trajectory.parquet")
                pq.write_table(sample_video_timestamps_table, stage1_dir / "video_timestamps.parquet")

                # Process
                output_dir = Path(tmpdir) / "stage2"
                config = Stage2Config(urdf_path="/path/to/robot.urdf")
                processor = IKPostProcessor(config)
                processor.setup()
                processor.process_episode(stage1_dir, output_dir)

                # Check outputs
                assert (output_dir / "joints.parquet").exists()
                assert (output_dir / "metadata.json").exists()

                # Check parquet content
                joints_table = pq.read_table(output_dir / "joints.parquet")
                assert len(joints_table) == len(sample_video_timestamps_table)

                # Check metadata content
                with open(output_dir / "metadata.json") as f:
                    metadata = json.load(f)
                assert metadata["stage"] == 2
                assert "total_frames" in metadata
