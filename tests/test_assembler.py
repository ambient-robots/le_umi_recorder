"""
Tests for umi.pipeline.assembler module.

Tests dataset assembly from pipeline stage outputs.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from umi.pipeline.assembler import PipelineAssembler, assemble_session
from umi.pipeline.schemas import (
    AssemblyConfig,
    GripperState,
    JointState,
    gripper_states_to_table,
    joint_states_to_table,
)


class TestPipelineAssembler:
    """Tests for PipelineAssembler class."""

    @pytest.fixture
    def assembler(self, temp_output_dir):
        """Create an assembler instance."""
        config = AssemblyConfig(fps=60, repo_id="test/dataset")
        assembler = PipelineAssembler(config, temp_output_dir)
        return assembler

    def test_setup_creates_directories(self, assembler):
        """Test that setup creates required directories."""
        assembler.setup()

        assert (assembler.output_dir / "meta").exists()
        assert (assembler.output_dir / "data").exists()
        assert (assembler.output_dir / "videos").exists()

    def test_setup_continues_from_existing(self, assembler):
        """Test that setup continues from existing episodes file."""
        assembler.setup()

        # Create episodes.jsonl with 2 existing episodes
        episodes_file = assembler.output_dir / "meta" / "episodes.jsonl"
        with open(episodes_file, "w") as f:
            f.write('{"episode_index": 0}\n')
            f.write('{"episode_index": 1}\n')

        # Re-setup
        assembler2 = PipelineAssembler(assembler.config, assembler.output_dir)
        assembler2.setup()

        assert assembler2.episode_index == 2

    def test_build_episode_data_shape(self, assembler, sample_joints_table, sample_gripper_table, sample_video_timestamps_table):
        """Test that built episode data has correct shape."""
        assembler.setup()

        episode_data = assembler._build_episode_data(
            sample_joints_table,
            sample_gripper_table,
            sample_video_timestamps_table,
            episode_index=0,
        )

        assert len(episode_data) == len(sample_joints_table)

        # Check action shape (7D: 6 joints + 1 gripper)
        actions = episode_data["action"].to_pylist()
        assert len(actions[0]) == 7

        # Check observation.state shape (7D: 6 joints + 1 gripper)
        obs_states = episode_data["observation.state"].to_pylist()
        assert len(obs_states[0]) == 7

    def test_build_episode_data_gripper_scaling(self, assembler, sample_joints_table, sample_gripper_states, sample_video_timestamps_table):
        """Test that gripper values are scaled from 0-100 to 0-1."""
        assembler.setup()

        # Create gripper states with known values
        gripper_states = [
            GripperState(
                frame_index=i,
                timestamp_ns=1000000000 + i * 16666667,
                measured_state=100.0,  # Should become 1.0
                commanded_state=50.0,  # Should become 0.5
                confidence=0.9,
            )
            for i in range(len(sample_joints_table))
        ]
        gripper_table = gripper_states_to_table(gripper_states)

        episode_data = assembler._build_episode_data(
            sample_joints_table,
            gripper_table,
            sample_video_timestamps_table,
            episode_index=0,
        )

        # Check gripper values in observation.state (last element)
        obs_states = episode_data["observation.state"].to_pylist()
        assert obs_states[0][6] == pytest.approx(1.0, abs=0.01)  # measured_state scaled

        # Check gripper values in action (last element)
        actions = episode_data["action"].to_pylist()
        assert actions[0][6] == pytest.approx(0.5, abs=0.01)  # commanded_state scaled

    def test_build_episode_data_frame_indices(self, assembler, sample_joints_table, sample_gripper_table, sample_video_timestamps_table):
        """Test that frame indices are sequential."""
        assembler.setup()

        episode_data = assembler._build_episode_data(
            sample_joints_table,
            sample_gripper_table,
            sample_video_timestamps_table,
            episode_index=0,
        )

        frame_indices = episode_data["frame_index"].to_pylist()
        assert frame_indices == list(range(len(sample_joints_table)))

    def test_build_episode_data_episode_index(self, assembler, sample_joints_table, sample_gripper_table, sample_video_timestamps_table):
        """Test that episode indices are set correctly."""
        assembler.setup()

        episode_data = assembler._build_episode_data(
            sample_joints_table,
            sample_gripper_table,
            sample_video_timestamps_table,
            episode_index=42,
        )

        episode_indices = episode_data["episode_index"].to_pylist()
        assert all(idx == 42 for idx in episode_indices)

    def test_compute_stats_mean_std(self, assembler, sample_joints_table, sample_gripper_table, sample_video_timestamps_table):
        """Test that statistics are computed correctly."""
        assembler.setup()

        # Build some episode data
        episode_data = assembler._build_episode_data(
            sample_joints_table,
            sample_gripper_table,
            sample_video_timestamps_table,
            episode_index=0,
        )

        # Update stats
        assembler._update_stats(episode_data)

        # Compute stats
        stats = assembler._compute_stats()

        assert "action" in stats
        assert "observation.state" in stats

        # Check that mean and std have 7 elements
        assert len(stats["action"]["mean"]) == 7
        assert len(stats["action"]["std"]) == 7
        assert len(stats["action"]["min"]) == 7
        assert len(stats["action"]["max"]) == 7


class TestAssembleEpisode:
    """Tests for assembling complete episodes."""

    def test_assemble_episode_creates_files(self, temp_episode_dir, temp_output_dir):
        """Test that assemble_episode creates expected output files."""
        config = AssemblyConfig(fps=60)
        assembler = PipelineAssembler(config, temp_output_dir)
        assembler.setup()

        # Need to create proper stage2/joints.parquet
        # The temp_episode_dir fixture should have this
        num_frames = assembler.assemble_episode(temp_episode_dir, camera_name="wrist")

        # Check files were created
        assert (temp_output_dir / "data" / "episode_000000.parquet").exists()
        assert (temp_output_dir / "videos" / "observation.images.wrist" / "episode_000000.mp4").exists()

        # Check episodes.jsonl was updated
        episodes_file = temp_output_dir / "meta" / "episodes.jsonl"
        assert episodes_file.exists()
        with open(episodes_file) as f:
            episode_meta = json.loads(f.readline())
        assert episode_meta["episode_index"] == 0
        assert episode_meta["num_frames"] == num_frames

    def test_assemble_episode_missing_files_raises(self, temp_output_dir):
        """Test that missing required files raise error."""
        config = AssemblyConfig(fps=60)
        assembler = PipelineAssembler(config, temp_output_dir)
        assembler.setup()

        with tempfile.TemporaryDirectory() as empty_dir:
            empty_episode = Path(empty_dir) / "episode_000"
            empty_episode.mkdir(parents=True)
            (empty_episode / "stage1").mkdir()
            (empty_episode / "stage2").mkdir()

            with pytest.raises(FileNotFoundError):
                assembler.assemble_episode(empty_episode)


class TestFinalize:
    """Tests for dataset finalization."""

    def test_finalize_creates_meta_files(self, temp_episode_dir, temp_output_dir):
        """Test that finalize creates info.json and stats.json."""
        config = AssemblyConfig(fps=60, repo_id="test/my_dataset")
        assembler = PipelineAssembler(config, temp_output_dir)
        assembler.setup()

        # Assemble one episode
        assembler.assemble_episode(temp_episode_dir)

        # Finalize
        assembler.finalize(camera_name="wrist")

        # Check files
        assert (temp_output_dir / "meta" / "info.json").exists()
        assert (temp_output_dir / "meta" / "stats.json").exists()

    def test_finalize_info_json_content(self, temp_episode_dir, temp_output_dir):
        """Test info.json has expected content."""
        config = AssemblyConfig(fps=60, repo_id="test/my_dataset")
        assembler = PipelineAssembler(config, temp_output_dir)
        assembler.setup()
        assembler.assemble_episode(temp_episode_dir)
        assembler.finalize(camera_name="wrist")

        with open(temp_output_dir / "meta" / "info.json") as f:
            info = json.load(f)

        assert info["codebase_version"] == "v2.1"
        assert info["robot_type"] == "umi"
        assert info["fps"] == 60
        assert info["total_episodes"] == 1
        assert "action" in info["features"]
        assert "observation.state" in info["features"]
        assert "observation.images.wrist" in info["features"]
        assert info["repo_id"] == "test/my_dataset"

    def test_finalize_action_feature_shape(self, temp_episode_dir, temp_output_dir):
        """Test action feature has correct shape (7D)."""
        config = AssemblyConfig(fps=60)
        assembler = PipelineAssembler(config, temp_output_dir)
        assembler.setup()
        assembler.assemble_episode(temp_episode_dir)
        assembler.finalize()

        with open(temp_output_dir / "meta" / "info.json") as f:
            info = json.load(f)

        assert info["features"]["action"]["shape"] == [7]
        assert info["features"]["action"]["dtype"] == "float32"

    def test_finalize_stats_json_content(self, temp_episode_dir, temp_output_dir):
        """Test stats.json has expected structure."""
        config = AssemblyConfig(fps=60)
        assembler = PipelineAssembler(config, temp_output_dir)
        assembler.setup()
        assembler.assemble_episode(temp_episode_dir)
        assembler.finalize()

        with open(temp_output_dir / "meta" / "stats.json") as f:
            stats = json.load(f)

        assert "action" in stats
        assert "observation.state" in stats

        # Each should have mean, std, min, max
        for key in ["action", "observation.state"]:
            assert "mean" in stats[key]
            assert "std" in stats[key]
            assert "min" in stats[key]
            assert "max" in stats[key]


class TestAssembleSession:
    """Tests for session-level assembly."""

    def test_assemble_session_multiple_episodes(self, temp_output_dir):
        """Test assembling a session with multiple episodes."""
        # Create a session directory with multiple episodes
        with tempfile.TemporaryDirectory() as session_dir:
            session_path = Path(session_dir)

            # Create 3 episode directories
            for ep_idx in range(3):
                ep_dir = session_path / f"episode_{ep_idx:03d}"

                # Stage 1
                stage1 = ep_dir / "stage1"
                stage1.mkdir(parents=True)
                (stage1 / "raw_video.mp4").write_bytes(b"video")

                # Create minimal parquet files
                ts_data = {
                    "frame_index": list(range(10)),
                    "timestamp_ns": [1000000000 + i * 16666667 for i in range(10)],
                    "closest_trajectory_idx": list(range(10)),
                }
                from umi.pipeline.schemas import VIDEO_TIMESTAMPS_SCHEMA
                ts_table = pa.Table.from_pydict(ts_data, schema=VIDEO_TIMESTAMPS_SCHEMA)
                pq.write_table(ts_table, stage1 / "video_timestamps.parquet")

                # Stage 2
                stage2 = ep_dir / "stage2"
                stage2.mkdir(parents=True)
                joints = [
                    JointState(
                        timestamp_ns=1000000000 + i * 16666667,
                        frame_index=i,
                        shoulder_pan=0.0,
                        shoulder_lift=0.0,
                        elbow_flex=0.0,
                        elbow_roll=0.0,
                        wrist_flex=0.0,
                        wrist_roll=0.0,
                        ik_success=True,
                        ik_error=0.0,
                    )
                    for i in range(10)
                ]
                pq.write_table(joint_states_to_table(joints), stage2 / "joints.parquet")

                # Stage 3
                stage3 = ep_dir / "stage3"
                stage3.mkdir(parents=True)
                with open(stage3 / "task_prompt.json", "w") as f:
                    json.dump({"task_description": f"Episode {ep_idx} task"}, f)

                # Stage 4
                stage4 = ep_dir / "stage4"
                stage4.mkdir(parents=True)
                grippers = [
                    GripperState(
                        frame_index=i,
                        timestamp_ns=1000000000 + i * 16666667,
                        measured_state=50.0,
                        commanded_state=50.0,
                        confidence=0.9,
                    )
                    for i in range(10)
                ]
                pq.write_table(gripper_states_to_table(grippers), stage4 / "gripper_states.parquet")

            # Assemble session
            config = AssemblyConfig(fps=60)
            assemble_session(session_path, temp_output_dir, config)

            # Check output
            assert (temp_output_dir / "meta" / "info.json").exists()
            with open(temp_output_dir / "meta" / "info.json") as f:
                info = json.load(f)
            assert info["total_episodes"] == 3

            # Check all episode files exist
            for i in range(3):
                assert (temp_output_dir / "data" / f"episode_{i:06d}.parquet").exists()

    def test_assemble_session_no_episodes(self, temp_output_dir):
        """Test that empty session directory is handled gracefully."""
        with tempfile.TemporaryDirectory() as session_dir:
            # Should print message but not crash
            config = AssemblyConfig(fps=60)
            assemble_session(Path(session_dir), temp_output_dir, config)

            # Output dir should not have been set up (no episodes to process)
            assert not (temp_output_dir / "meta" / "info.json").exists()
