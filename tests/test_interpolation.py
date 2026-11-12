"""
Tests for umi.pipeline.interpolation module.

These are pure function tests with no external dependencies.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from umi.pipeline.interpolation import (
    interpolate_position,
    interpolate_quaternion,
    interpolate_scalar,
    find_bracketing_indices,
    interpolate_trajectory_to_video,
    compute_video_trajectory_alignment,
    resample_trajectory,
    estimate_trajectory_fps,
    validate_sync_quality,
)
from umi.pipeline.schemas import VideoTimestamp


class TestInterpolatePosition:
    """Tests for linear position interpolation."""

    def test_interpolate_position_midpoint(self):
        """Test interpolation at midpoint returns average."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([2.0, 4.0, 6.0])

        result = interpolate_position(0.5, 0.0, p0, 1.0, p1)

        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_interpolate_position_at_start(self):
        """Test interpolation at t=t0 returns p0."""
        p0 = np.array([1.0, 2.0, 3.0])
        p1 = np.array([4.0, 5.0, 6.0])

        result = interpolate_position(0.0, 0.0, p0, 1.0, p1)

        np.testing.assert_array_almost_equal(result, p0)

    def test_interpolate_position_at_end(self):
        """Test interpolation at t=t1 returns p1."""
        p0 = np.array([1.0, 2.0, 3.0])
        p1 = np.array([4.0, 5.0, 6.0])

        result = interpolate_position(1.0, 0.0, p0, 1.0, p1)

        np.testing.assert_array_almost_equal(result, p1)

    def test_interpolate_position_same_time(self):
        """Test interpolation when t0 == t1 returns p0."""
        p0 = np.array([1.0, 2.0, 3.0])
        p1 = np.array([4.0, 5.0, 6.0])

        result = interpolate_position(0.5, 0.5, p0, 0.5, p1)

        np.testing.assert_array_almost_equal(result, p0)

    def test_interpolate_position_quarter(self):
        """Test interpolation at 25% returns correct value."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([4.0, 8.0, 12.0])

        result = interpolate_position(0.25, 0.0, p0, 1.0, p1)

        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])


class TestInterpolateQuaternion:
    """Tests for SLERP quaternion interpolation."""

    def test_interpolate_quaternion_identity(self):
        """Test interpolation between identical quaternions."""
        q = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion

        result = interpolate_quaternion(0.5, 0.0, q, 1.0, q)

        np.testing.assert_array_almost_equal(result, q)

    def test_interpolate_quaternion_at_start(self):
        """Test interpolation at t=t0 returns q0."""
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        q1 = np.array([0.0, 0.0, 0.707, 0.707])  # 90° around Z

        result = interpolate_quaternion(0.0, 0.0, q0, 1.0, q1)

        np.testing.assert_array_almost_equal(result, q0, decimal=3)

    def test_interpolate_quaternion_at_end(self):
        """Test interpolation at t=t1 returns q1."""
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        q1 = np.array([0.0, 0.0, 0.707, 0.707])  # 90° around Z

        result = interpolate_quaternion(1.0, 0.0, q0, 1.0, q1)

        np.testing.assert_array_almost_equal(result, q1, decimal=3)

    def test_interpolate_quaternion_midpoint(self):
        """Test SLERP at midpoint gives 45° rotation."""
        q0 = np.array([0.0, 0.0, 0.0, 1.0])  # Identity
        # 90° around Z
        q1 = Rotation.from_euler('z', 90, degrees=True).as_quat()

        result = interpolate_quaternion(0.5, 0.0, q0, 1.0, q1)

        # Expected: 45° around Z
        expected = Rotation.from_euler('z', 45, degrees=True).as_quat()
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_interpolate_quaternion_same_time(self):
        """Test interpolation when t0 == t1 returns q0."""
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        q1 = np.array([0.0, 0.0, 0.707, 0.707])

        result = interpolate_quaternion(0.5, 0.5, q0, 0.5, q1)

        np.testing.assert_array_almost_equal(result, q0)


class TestInterpolateScalar:
    """Tests for linear scalar interpolation."""

    def test_interpolate_scalar_midpoint(self):
        """Test scalar interpolation at midpoint."""
        result = interpolate_scalar(0.5, 0.0, 0.0, 1.0, 100.0)
        assert result == pytest.approx(50.0)

    def test_interpolate_scalar_at_start(self):
        """Test scalar interpolation at start."""
        result = interpolate_scalar(0.0, 0.0, 10.0, 1.0, 20.0)
        assert result == pytest.approx(10.0)

    def test_interpolate_scalar_at_end(self):
        """Test scalar interpolation at end."""
        result = interpolate_scalar(1.0, 0.0, 10.0, 1.0, 20.0)
        assert result == pytest.approx(20.0)

    def test_interpolate_scalar_same_time(self):
        """Test scalar interpolation when t0 == t1."""
        result = interpolate_scalar(0.5, 0.5, 10.0, 0.5, 20.0)
        assert result == pytest.approx(10.0)


class TestFindBracketingIndices:
    """Tests for finding bracketing indices in sorted arrays."""

    def test_find_bracketing_middle(self):
        """Test finding brackets in the middle of array."""
        timestamps = np.array([0, 100, 200, 300, 400])

        before, after = find_bracketing_indices(timestamps, 150)

        assert before == 1
        assert after == 2

    def test_find_bracketing_at_start(self):
        """Test finding brackets at array start."""
        timestamps = np.array([100, 200, 300, 400, 500])

        before, after = find_bracketing_indices(timestamps, 50)

        assert before == 0
        assert after == 0

    def test_find_bracketing_at_end(self):
        """Test finding brackets beyond array end."""
        timestamps = np.array([100, 200, 300, 400, 500])

        before, after = find_bracketing_indices(timestamps, 600)

        assert before == 4
        assert after == 4

    def test_find_bracketing_exact_match(self):
        """Test when target exactly matches a timestamp."""
        timestamps = np.array([100, 200, 300, 400, 500])

        before, after = find_bracketing_indices(timestamps, 200)

        # Should bracket with 200 being the "after"
        assert before == 0
        assert after == 1


class TestValidateSyncQuality:
    """Tests for synchronization quality validation."""

    def test_validate_sync_quality_good_sync(self):
        """Test sync validation with well-aligned timestamps."""
        traj_ts = np.array([0, 20, 40, 60, 80, 100]) * 1_000_000  # 0-100ms
        video_ts = np.array([0, 17, 33, 50, 67, 83, 100]) * 1_000_000

        is_valid, max_drift, msg = validate_sync_quality(traj_ts, video_ts)

        assert is_valid
        assert max_drift == pytest.approx(0.0, abs=0.1)
        assert "OK" in msg

    def test_validate_sync_quality_drift(self):
        """Test sync validation with drift."""
        traj_ts = np.array([0, 20, 40, 60, 80, 100]) * 1_000_000
        video_ts = np.array([10, 27, 43, 60, 77, 93, 110]) * 1_000_000  # 10ms offset

        is_valid, max_drift, msg = validate_sync_quality(traj_ts, video_ts, max_drift_ms=5.0)

        assert not is_valid
        assert max_drift == pytest.approx(10.0, abs=0.1)
        assert "exceeds" in msg.lower()

    def test_validate_sync_quality_no_overlap(self):
        """Test sync validation with no temporal overlap."""
        traj_ts = np.array([0, 20, 40]) * 1_000_000
        video_ts = np.array([100, 120, 140]) * 1_000_000

        is_valid, max_drift, msg = validate_sync_quality(traj_ts, video_ts)

        assert not is_valid
        assert max_drift == float("inf")
        assert "overlap" in msg.lower()

    def test_validate_sync_quality_empty_arrays(self):
        """Test sync validation with empty arrays."""
        is_valid, max_drift, msg = validate_sync_quality(np.array([]), np.array([]))

        assert not is_valid
        assert "empty" in msg.lower()


class TestEstimateTrajectoryFps:
    """Tests for FPS estimation."""

    def test_estimate_fps_50hz(self, sample_trajectory_table):
        """Test FPS estimation for 50Hz trajectory."""
        fps = estimate_trajectory_fps(sample_trajectory_table)

        assert fps == pytest.approx(50.0, rel=0.01)

    def test_estimate_fps_single_frame(self):
        """Test FPS estimation with single frame returns 0."""
        import pyarrow as pa
        from umi.pipeline.schemas import TRAJECTORY_SCHEMA

        # Create single-frame table
        data = {
            "timestamp_ns": [1000000000],
            "frame_index": [0],
            "position_x": [0.0],
            "position_y": [0.0],
            "position_z": [0.0],
            "orientation_x": [0.0],
            "orientation_y": [0.0],
            "orientation_z": [0.0],
            "orientation_w": [1.0],
            "button_trigger": [0.0],
            "button_grip": [0.0],
            "controller_hand": ["right"],
        }
        table = pa.Table.from_pydict(data, schema=TRAJECTORY_SCHEMA)

        fps = estimate_trajectory_fps(table)

        assert fps == 0.0


class TestInterpolateTrajectoryToVideo:
    """Tests for full trajectory-to-video interpolation."""

    def test_interpolate_trajectory_length(
        self, sample_trajectory_table, sample_video_timestamps
    ):
        """Test that output length matches video timestamps."""
        result = interpolate_trajectory_to_video(
            sample_trajectory_table, sample_video_timestamps
        )

        assert len(result) == len(sample_video_timestamps)

    def test_interpolate_trajectory_timestamps_match(
        self, sample_trajectory_table, sample_video_timestamps
    ):
        """Test that output timestamps match video timestamps."""
        result = interpolate_trajectory_to_video(
            sample_trajectory_table, sample_video_timestamps
        )

        for frame, video_ts in zip(result, sample_video_timestamps):
            assert frame.timestamp_ns == video_ts.timestamp_ns
            assert frame.frame_index == video_ts.frame_index

    def test_interpolate_trajectory_preserves_hand(
        self, sample_trajectory_table, sample_video_timestamps
    ):
        """Test that controller hand is preserved."""
        result = interpolate_trajectory_to_video(
            sample_trajectory_table, sample_video_timestamps
        )

        for frame in result:
            assert frame.controller_hand == "right"

    def test_interpolate_trajectory_quaternion_normalized(
        self, sample_trajectory_table, sample_video_timestamps
    ):
        """Test that interpolated quaternions are approximately normalized."""
        result = interpolate_trajectory_to_video(
            sample_trajectory_table, sample_video_timestamps
        )

        for frame in result:
            quat = np.array([
                frame.orientation_x,
                frame.orientation_y,
                frame.orientation_z,
                frame.orientation_w,
            ])
            norm = np.linalg.norm(quat)
            assert norm == pytest.approx(1.0, abs=0.01)


class TestComputeVideoTrajectoryAlignment:
    """Tests for finding closest trajectory frames."""

    def test_alignment_length(
        self, trajectory_timestamps_array, video_timestamps_array
    ):
        """Test that alignment output matches video length."""
        result = compute_video_trajectory_alignment(
            trajectory_timestamps_array, video_timestamps_array
        )

        assert len(result) == len(video_timestamps_array)

    def test_alignment_valid_indices(
        self, trajectory_timestamps_array, video_timestamps_array
    ):
        """Test that all indices are valid."""
        result = compute_video_trajectory_alignment(
            trajectory_timestamps_array, video_timestamps_array
        )

        for idx in result:
            assert 0 <= idx < len(trajectory_timestamps_array)

    def test_alignment_increasing(
        self, trajectory_timestamps_array, video_timestamps_array
    ):
        """Test that alignment indices are non-decreasing."""
        result = compute_video_trajectory_alignment(
            trajectory_timestamps_array, video_timestamps_array
        )

        for i in range(1, len(result)):
            assert result[i] >= result[i - 1]


class TestResampleTrajectory:
    """Tests for trajectory resampling."""

    def test_resample_to_higher_fps(self, sample_trajectory_table):
        """Test resampling from 50Hz to 60Hz."""
        result = resample_trajectory(sample_trajectory_table, target_fps=60.0, source_fps=50.0)

        # Should have more frames
        assert len(result) > len(sample_trajectory_table)

        # Verify timestamps are evenly spaced (using linspace-generated intervals)
        timestamps = result["timestamp_ns"].to_numpy()
        intervals = np.diff(timestamps)

        # Check that intervals are consistent with each other (uniform spacing)
        mean_interval = np.mean(intervals)
        np.testing.assert_array_almost_equal(
            intervals, np.full_like(intervals, mean_interval), decimal=-4
        )

        # Check that mean interval is approximately 60Hz
        expected_interval = 1e9 / 60  # 60Hz in nanoseconds
        assert mean_interval == pytest.approx(expected_interval, rel=0.01)

    def test_resample_to_lower_fps(self, sample_trajectory_table):
        """Test resampling from 50Hz to 30Hz."""
        result = resample_trajectory(sample_trajectory_table, target_fps=30.0, source_fps=50.0)

        # Should have fewer frames
        assert len(result) < len(sample_trajectory_table)
