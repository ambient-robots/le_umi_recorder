"""
Stage 3: Task Prompt Generation

Uses Gemini API to analyze episode videos and generate natural language
task descriptions suitable for robot policy training.

Input:
    - Stage 1 output: raw_video.mp4

Output:
    - task_prompt.json: Generated task description
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional

from .schemas import (
    Stage3Config,
    StageMetadata,
    TaskPrompt,
    PipelineConfig,
)
from .gemini_client import (
    GeminiClient,
    get_task_prompt,
    GENAI_AVAILABLE,
)


class TaskPromptGenerator:
    """
    Generates task descriptions from robot manipulation videos.

    Uses Gemini's video understanding capabilities to analyze
    the video and produce a concise task description.
    """

    def __init__(self, config: Stage3Config):
        """
        Initialize task prompt generator.

        Args:
            config: Stage 3 configuration
        """
        self.config = config
        self.client: Optional[GeminiClient] = None

    async def setup(self):
        """Initialize Gemini client."""
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )

        self.client = GeminiClient(
            model=self.config.model,
            api_key_env=self.config.api_key_env,
        )
        print(f"Gemini client initialized with model: {self.config.model}")

    async def generate_task_prompt(self, video_path: Path) -> TaskPrompt:
        """
        Generate task description for a video.

        Args:
            video_path: Path to episode video

        Returns:
            TaskPrompt with generated description
        """
        if self.client is None:
            await self.setup()

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        print(f"Analyzing video: {video_path.name}")

        # Get prompt template
        prompt = get_task_prompt(self.config.prompt_template)

        # Analyze video
        response = await self.client.analyze_video(
            video_path,
            prompt,
            json_response=True,
        )

        # Parse response
        data = self.client.parse_json_response(response)

        from datetime import datetime

        return TaskPrompt(
            task_description=data.get("task_description", "Unknown task"),
            confidence=float(data.get("confidence", 0.5)),
            model=self.config.model,
            generation_timestamp=datetime.now().isoformat(),
            video_summary=data.get("video_summary", ""),
            objects=data.get("objects", []),
            actions=data.get("actions", []),
        )

    async def process_episode(self, stage1_dir: Path, output_dir: Path) -> Path:
        """
        Process a single episode from Stage 1.

        Args:
            stage1_dir: Path to Stage 1 output directory
            output_dir: Output directory for Stage 3 results

        Returns:
            Path to output directory
        """
        stage1_dir = Path(stage1_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing episode: {stage1_dir}")

        # Find video file
        video_path = stage1_dir / "raw_video.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Generate task prompt
        task_prompt = await self.generate_task_prompt(video_path)

        print(f"  Task: {task_prompt.task_description}")
        print(f"  Confidence: {task_prompt.confidence:.2f}")

        # Save task prompt
        output_file = output_dir / "task_prompt.json"
        with open(output_file, "w") as f:
            json.dump(task_prompt.to_dict(), f, indent=2)

        # Save metadata
        metadata = StageMetadata(
            stage=3,
            version="1.0.0",
            extra={
                "model": self.config.model,
                "prompt_template": self.config.prompt_template,
                "video_path": str(video_path),
            },
        )
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        print(f"  Saved task prompt to: {output_file}")

        return output_dir

    async def cleanup(self):
        """Cleanup resources."""
        if self.client:
            await self.client.cleanup()


async def main():
    """CLI entry point for Stage 3 task prompt generation."""
    parser = argparse.ArgumentParser(
        description="Stage 3: Task Prompt Generation - Generate task descriptions with Gemini"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to Stage 1 output directory (or video file directly)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory (default: input/../stage3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Gemini model to use (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to pipeline config YAML file",
    )

    args = parser.parse_args()

    # Build config
    if args.config and args.config.exists():
        config = PipelineConfig.from_yaml(args.config)
        stage3_config = config.stage3
    else:
        stage3_config = Stage3Config(model=args.model)

    # Determine paths
    input_path = Path(args.input)

    # Check if input is video file or directory
    if input_path.suffix == ".mp4":
        # Direct video file
        video_path = input_path
        stage1_dir = input_path.parent
    else:
        # Directory
        stage1_dir = input_path
        video_path = stage1_dir / "raw_video.mp4"

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = stage1_dir.parent / "stage3"

    # Process
    generator = TaskPromptGenerator(stage3_config)
    try:
        await generator.setup()
        await generator.process_episode(stage1_dir, output_dir)
    finally:
        await generator.cleanup()

    print(f"\nStage 3 complete. Output: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
