"""
UMI Recording Pipeline

A staged pipeline for recording robot teleoperation data and generating
datasets compatible with Physical Intelligence PI 0.5/0.6 models.

Pipeline Stages:
    1. Raw Recording - Capture Quest trajectory (50Hz) + video (60Hz)
    2. IK Solver - Post-process with inverse kinematics
    3. Task Prompt - Generate task descriptions with Gemini
    4. Gripper Detection - Detect gripper states with Gemini
    5. Assembly - Combine into LeRobot v2.1 format

Usage:
    # Stage 1: Record raw data
    python -m umi.pipeline.stage1_recorder --output ./recordings/session_001

    # Stage 2: Post-process with IK
    python -m umi.pipeline.stage2_ik_solver --input ./recordings/.../stage1 --urdf robot.urdf

    # Stage 3: Generate task prompts (offline)
    python -m umi.pipeline.stage3_task_prompt --input ./recordings/.../stage1

    # Stage 4: Detect gripper states (offline)
    python -m umi.pipeline.stage4_gripper --input ./recordings/.../stage1

    # Assemble final dataset
    python -m umi.pipeline.assembler --input ./recordings/session --output ./datasets/my_dataset
"""

# Data schemas
from .schemas import (
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
    # Arrow schemas
    TRAJECTORY_SCHEMA,
    VIDEO_TIMESTAMPS_SCHEMA,
    JOINTS_SCHEMA,
    GRIPPER_STATES_SCHEMA,
    # Helper functions
    trajectory_frames_to_table,
    video_timestamps_to_table,
    joint_states_to_table,
    gripper_states_to_table,
)

# Stage implementations
from .stage1_recorder import RawRecorder
from .stage2_ik_solver import IKPostProcessor
from .stage3_task_prompt import TaskPromptGenerator
from .stage4_gripper import GripperDetector
from .assembler import PipelineAssembler, assemble_session

# Utilities
from .interpolation import (
    interpolate_trajectory_to_video,
    interpolate_position,
    interpolate_quaternion,
    compute_video_trajectory_alignment,
    resample_trajectory,
    estimate_trajectory_fps,
    validate_sync_quality,
)

# Gemini client (optional - requires google-generativeai)
try:
    from .gemini_client import (
        GeminiClient,
        GeminiAPIError,
        GeminiRateLimitError,
        get_task_prompt,
        get_gripper_prompt,
    )
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

__all__ = [
    # Data schemas
    "TrajectoryFrame",
    "VideoTimestamp",
    "JointState",
    "GripperState",
    "TaskPrompt",
    "StageMetadata",
    "PipelineConfig",
    "Stage1Config",
    "Stage2Config",
    "Stage3Config",
    "Stage4Config",
    "AssemblyConfig",
    # Arrow schemas
    "TRAJECTORY_SCHEMA",
    "VIDEO_TIMESTAMPS_SCHEMA",
    "JOINTS_SCHEMA",
    "GRIPPER_STATES_SCHEMA",
    # Helper functions
    "trajectory_frames_to_table",
    "video_timestamps_to_table",
    "joint_states_to_table",
    "gripper_states_to_table",
    # Stage implementations
    "RawRecorder",
    "IKPostProcessor",
    "TaskPromptGenerator",
    "GripperDetector",
    "PipelineAssembler",
    "assemble_session",
    # Interpolation utilities
    "interpolate_trajectory_to_video",
    "interpolate_position",
    "interpolate_quaternion",
    "compute_video_trajectory_alignment",
    "resample_trajectory",
    "estimate_trajectory_fps",
    "validate_sync_quality",
]

# Add Gemini exports if available
if _GEMINI_AVAILABLE:
    __all__.extend([
        "GeminiClient",
        "GeminiAPIError",
        "GeminiRateLimitError",
        "get_task_prompt",
        "get_gripper_prompt",
    ])
