#!/usr/bin/env python3
"""
GVL (General Vision Language) Analysis Tool

Analyze video frames with AI models (GPT-4o, Gemini Pro, Gemini Flash) to evaluate task completion.

Examples:
    # Basic usage with default settings
    python gvl.py

    # Single model with custom settings
    python gvl.py --video data/experimental/gvl/arm_video_2.mp4 --model gemini-flash --interval 15

    # Multiple models comparison
    python gvl.py --model gpt gemini-flash --interval 15

    # All models comparison
    python gvl.py --model all --interval 30

    # Chess analysis with multiple models, no shuffling
    python gvl.py --task "Moving knight to checkmate position" --model gpt gemini-pro --no-shuffle

    # Quick analysis with all models
    python gvl.py --model all --interval 60 --task "Picking up the red cup"

API Keys Required:
    - For GPT: Set OPENAI_API_KEY environment variable
    - For Gemini: Set GEMINI_API_KEY environment variable
"""

import argparse
import base64
import io
import os
import random
import re
import sys
import time
import urllib.request
from typing import Any, Literal

import cv2
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import openai
from PIL import Image, ImageDraw, ImageFont

# API Cost tracking (costs in USD)
API_COSTS = {
    "gpt": {
        "input_cost_per_1k_tokens": 0.0025,  # GPT-4o vision input cost
        "output_cost_per_1k_tokens": 0.01,  # GPT-4o output cost
        "estimated_input_tokens_per_image": 765,  # Average for images
        "estimated_output_tokens_per_response": 50,  # Average response length
    },
    "gemini-pro": {
        "input_cost_per_1k_tokens": 0.00125,  # Gemini 1.5 Pro input cost
        "output_cost_per_1k_tokens": 0.005,  # Gemini 1.5 Pro output cost
        "estimated_input_tokens_per_image": 258,  # Gemini image token count
        "estimated_output_tokens_per_response": 50,
    },
    "gemini-flash": {
        "input_cost_per_1k_tokens": 0.000075,  # Gemini 2.0 Flash input cost
        "output_cost_per_1k_tokens": 0.0003,  # Gemini 2.0 Flash output cost
        "estimated_input_tokens_per_image": 258,  # Gemini image token count
        "estimated_output_tokens_per_response": 50,
    },
}

# HuggingFace URLs
context_video_1_url = "https://huggingface.co/datasets/dopaul/game_v1/resolve/main/videos/chunk-000/observation.images.context/episode_000000.mp4"
context_video_2_url = "https://huggingface.co/datasets/dopaul/game_v1/resolve/main/videos/chunk-000/observation.images.context/episode_000001.mp4"
context_video_3_url = "https://huggingface.co/datasets/dopaul/game_v9/resolve/main/videos/chunk-000/observation.images.context/episode_000000.mp4"
context_video_4_url = "https://huggingface.co/datasets/dopaul/game_v9/resolve/main/videos/chunk-000/observation.images.context/episode_000082.mp4"
arm_video_1_url = "https://huggingface.co/datasets/dopaul/game_v1/resolve/main/videos/chunk-000/observation.images.arm/episode_000000.mp4"
arm_video_2_url = "https://huggingface.co/datasets/dopaul/game_v1/resolve/main/videos/chunk-000/observation.images.arm/episode_000001.mp4"
arm_video_3_url = "https://huggingface.co/datasets/dopaul/game_v9/resolve/main/videos/chunk-000/observation.images.arm/episode_000000.mp4"
arm_video_4_url = "https://huggingface.co/datasets/dopaul/game_v9/resolve/main/videos/chunk-000/observation.images.arm/episode_000082.mp4"

# Local paths
os.makedirs("artifacts/gvl", exist_ok=True)
os.makedirs("data/experimental/gvl", exist_ok=True)
context_video_1_path = "data/experimental/gvl/context_video_1.mp4"
context_video_2_path = "data/experimental/gvl/context_video_2.mp4"
context_video_3_path = "data/experimental/gvl/context_video_3.mp4"
context_video_4_path = "data/experimental/gvl/context_video_4.mp4"
arm_video_1_path = "data/experimental/gvl/arm_video_1.mp4"
arm_video_2_path = "data/experimental/gvl/arm_video_2.mp4"
arm_video_3_path = "data/experimental/gvl/arm_video_3.mp4"
arm_video_4_path = "data/experimental/gvl/arm_video_4.mp4"


def download_video(url, local_path):
    """Download video from URL to local path if it doesn't exist."""
    if not os.path.exists(local_path):
        # Security: Validate URL to ensure it's HTTPS and from trusted domain
        if not url.startswith("https://"):
            raise ValueError(f"Only HTTPS URLs are allowed, got: {url}")
        if not any(domain in url for domain in ["huggingface.co", "hf.co"]):
            raise ValueError(f"Only trusted domains are allowed, got: {url}")

        print(f"Downloading {url} to {local_path}...")
        try:
            urllib.request.urlretrieve(url, local_path)  # nosec B310
            print(f"Successfully downloaded {local_path}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    else:
        print(f"{local_path} already exists, skipping download")


# Download videos if not already downloaded
download_video(context_video_1_url, context_video_1_path)
download_video(context_video_2_url, context_video_2_path)
download_video(context_video_3_url, context_video_3_path)
download_video(context_video_4_url, context_video_4_path)
download_video(arm_video_1_url, arm_video_1_path)
download_video(arm_video_2_url, arm_video_2_path)
download_video(arm_video_3_url, arm_video_3_path)
download_video(arm_video_4_url, arm_video_4_path)


def extract_every_nth_frame(video_path: str, n: int = 30) -> list[Any]:
    """Extract every nth frame from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % n == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert OpenCV frame to base64 string."""
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def estimate_api_cost(model: str, num_images: int, context_images: int = 0) -> float:
    """Estimate API cost based on model and number of images."""
    if model not in API_COSTS:
        return 0.0

    costs = API_COSTS[model]

    # Calculate input tokens (text + images)
    text_tokens_per_request = 200  # System prompt + user prompt
    image_tokens_per_request = costs["estimated_input_tokens_per_image"]

    # For conversational context, we accumulate previous images
    total_input_tokens = 0
    for i in range(num_images):
        current_context_images = i + context_images  # Previous images in context
        current_request_tokens = text_tokens_per_request + (image_tokens_per_request * (current_context_images + 1))
        total_input_tokens += current_request_tokens

    # Calculate output tokens
    total_output_tokens = num_images * costs["estimated_output_tokens_per_response"]

    # Calculate total cost
    input_cost = (total_input_tokens / 1000) * costs["input_cost_per_1k_tokens"]
    output_cost = (total_output_tokens / 1000) * costs["output_cost_per_1k_tokens"]

    return input_cost + output_cost


def call_model_api(
    model: Literal["gpt", "gemini-pro", "gemini-flash"],
    messages: list[dict],
    system_prompt: str,
    image_b64: str,
    prompt_text: str,
) -> tuple[str, float]:
    """Call the specified model API and return response content and timing."""
    start_time = time.time()

    try:
        if model == "gpt":
            # OpenAI GPT-4o
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,  # type: ignore
                max_tokens=150,
            )
            response_content = response.choices[0].message.content or ""

        elif model in ["gemini-pro", "gemini-flash"]:
            # Google Gemini
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            genai.configure(api_key=api_key)  # type: ignore

            model_name = "gemini-2.0-flash-exp" if model == "gemini-flash" else "gemini-1.5-pro"
            gemini_model = genai.GenerativeModel(model_name)  # type: ignore

            # Convert base64 to PIL Image for Gemini
            image_data = base64.b64decode(image_b64)
            image_pil = Image.open(io.BytesIO(image_data))

            # Create Gemini prompt
            full_prompt = f"{system_prompt}\n\n{prompt_text}"

            response = gemini_model.generate_content([full_prompt, image_pil])
            response_content = response.text

        else:
            raise ValueError(f"Unsupported model: {model}")

    except Exception as e:
        end_time = time.time()
        api_time = end_time - start_time
        print(f"Error calling {model} API: {e}")
        return "", api_time

    end_time = time.time()
    api_time = end_time - start_time
    return response_content, api_time


def analyze_frames_with_gpt(
    video_path: str,
    task_description: str = "Moving the chess piece from the red to the blue circle",
    model: Literal["gpt", "gemini-pro", "gemini-flash"] = "gpt",
    shuffle: bool = False,
    frame_interval: int = 30,
) -> tuple[list[float], list[int] | None, list[float], float]:
    """Analyze frames from video with AI models for task completion."""
    # Extract frames
    frames = extract_every_nth_frame(video_path, frame_interval)
    if not frames:
        return [], None, [], 0.0

    # Calculate estimated cost
    estimated_cost = estimate_api_cost(model, len(frames))

    # Create shuffled order if requested
    original_order_mapping = None
    if shuffle:
        indices = list(range(len(frames)))
        shuffled_indices = [0] + random.sample(indices[1:], len(indices) - 1)
        original_order_mapping = shuffled_indices.copy()
        frames = [frames[i] for i in shuffled_indices]

    # Initialize conversation context
    conversation_history = []
    values = []
    api_times = []

    # Create the system prompt
    system_prompt = f"""
    You are an expert roboticist tasked to predict task completion
    percentages for frames of a robot for the task of {task_description}.
    The task completion percentages are between 0 and 100, where 100
    corresponds to full task completion. We provide several examples of
    the robot performing the task at various stages and their
    corresponding task completion percentages. Note that these frames are
    in random order, so please pay attention to the individual frames
    when reasoning about task completion percentage.

    For each frame, format your response as follow:
    Frame X: Frame Description: [brief description], Task Completion Percentage: Y%

    Focus on the key visual indicators that show progress toward {task_description}.
    """

    for i, frame in enumerate(frames):
        # Convert frame to base64
        frame_b64 = frame_to_base64(frame)

        # Build messages for this request
        messages = [{"role": "system", "content": system_prompt}]

        # Add previous context (images and responses)
        for prev_frame_b64, prev_response in conversation_history:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Previous frame:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{prev_frame_b64}"}},
                    ],
                }
            )  # type: ignore
            messages.append({"role": "assistant", "content": prev_response})

        # Add current frame
        if i == 0:
            frame_text = f"Initial robot scene for task: {task_description}. What is the task completion percentage for this frame?"
        else:
            frame_text = (
                f"Frame {i + 1} for task: {task_description}. What is the task completion percentage for this frame?"
            )

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": frame_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                ],
            }
        )  # type: ignore

        # Make API call with timing
        response_content, api_time = call_model_api(
            model=model, messages=messages, system_prompt=system_prompt, image_b64=frame_b64, prompt_text=frame_text
        )
        api_times.append(api_time)

        # Parse percentage from response
        try:
            if response_content is None:
                response_content = "Frame 1: Frame Description: unknown, Task Completion Percentage: 0%"

            # Extract percentage using regex
            percentage_match = re.search(r"Task Completion Percentage:\s*(\d+(?:\.\d+)?)%", response_content)
            if percentage_match:
                percentage = float(percentage_match.group(1))
                values.append(percentage)
            else:
                # Fallback: look for any number followed by %
                fallback_match = re.search(r"(\d+(?:\.\d+)?)%", response_content)
                if fallback_match:
                    percentage = float(fallback_match.group(1))
                    values.append(percentage)
                else:
                    print(f"Could not extract percentage from response: {response_content}")
                    values.append(0.0)  # Default value

            # Add to conversation history
            conversation_history.append((frame_b64, response_content))

            # Show original frame ID if shuffled
            if original_order_mapping:
                original_frame_id = original_order_mapping[i]
                print(f"Frame {i + 1} (original #{original_frame_id}): {values[-1]}% (API time: {api_times[-1]:.2f}s)")
            else:
                print(f"Frame {i + 1}: {values[-1]}% (API time: {api_times[-1]:.2f}s)")

        except Exception as e:
            print(f"Error parsing response for frame {i + 1}: {e}")
            print(f"Response: {response_content}")
            values.append(0.0)  # Default value
            conversation_history.append((frame_b64, response_content or ""))

    return values, original_order_mapping, api_times, estimated_cost


def create_multimodel_video_with_scores(
    frames: list[np.ndarray], all_model_scores: dict, output_path: str, model_colors: dict, fps: float = 4.0
) -> None:
    """Create a video from frames with multi-model score plot in corner."""
    if not frames or not all_model_scores:
        print("No frames or scores provided")
        return

    plt.switch_backend("Agg")
    num_frames = len(frames)

    # Try different codecs for better compatibility
    codecs_to_try = [
        cv2.VideoWriter.fourcc(*"avc1"),  # H.264  # type: ignore
        cv2.VideoWriter.fourcc(*"mp4v"),  # MPEG-4  # type: ignore
        cv2.VideoWriter.fourcc(*"XVID"),  # XVID  # type: ignore
    ]

    video_writer = None
    successful_codec = None

    for i, frame in enumerate(frames):
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Create a higher resolution plot for the current frame with all models
        fig, ax = plt.subplots(figsize=(4, 2.5))  # Slightly larger figure

        # Plot each model's scores up to current frame
        for model_name, scores in all_model_scores.items():
            current_scores = scores[: i + 1]
            current_indices = list(range(len(current_scores)))
            color = model_colors.get(model_name, "gray")

            ax.plot(current_indices, current_scores, color=color, linewidth=3, label=model_name)  # Thicker lines
            ax.scatter(current_indices, current_scores, color=color, s=30, zorder=5)  # Larger dots
            if i < len(scores):
                ax.scatter([i], [scores[i]], color=color, s=60, zorder=6)  # Larger current point

        ax.set_xlim(0, num_frames - 1)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Frame", fontsize=10)  # Larger font
        ax.set_ylabel("Completion %", fontsize=10)  # Larger font
        ax.set_title(f"Progress (Frame {i + 1})", fontsize=12)  # Larger font
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)  # Larger tick labels

        # Add legend if multiple models
        if len(all_model_scores) > 1:
            ax.legend(fontsize=8, loc="upper left")  # Larger legend font

        # Convert plot to numpy array with higher DPI
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")  # Doubled DPI for better quality
        buf.seek(0)
        plot_pil = Image.open(buf)

        # Resize plot to fit in corner
        plot_size = (750, 480)
        plot_pil = plot_pil.resize(plot_size, Image.Resampling.LANCZOS)

        # Use the original frame as base
        min_width, min_height = 480, 360
        if frame_pil.width < min_width or frame_pil.height < min_height:
            scale_factor = max(min_width / frame_pil.width, min_height / frame_pil.height)
            new_width = int(frame_pil.width * scale_factor)
            new_height = int(frame_pil.height * scale_factor)
            frame_pil = frame_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create canvas with overlay
        canvas = frame_pil.copy()

        # Add semi-transparent background for plot (bottom left)
        plot_bg = Image.new("RGBA", plot_size, (255, 255, 255, 200))
        plot_x = 10
        plot_y = canvas.height - plot_size[1] - 10

        canvas.paste(plot_bg, (plot_x, plot_y), plot_bg)
        canvas.paste(plot_pil, (plot_x, plot_y), plot_pil)

        # Add frame number text overlay (top right)
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("Arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

        text = f"Frame {i + 1}/{len(frames)}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Text background (top right)
        text_bg_x = canvas.width - text_width - 20
        text_bg_y = 10
        draw.rectangle(
            [text_bg_x - 5, text_bg_y - 2, text_bg_x + text_width + 5, text_bg_y + text_height + 2], fill=(0, 0, 0, 150)
        )
        draw.text((text_bg_x, text_bg_y), text, fill="white", font=font)

        # Convert PIL image back to OpenCV format (BGR)
        canvas_cv = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

        # Initialize video writer with first frame dimensions
        if video_writer is None:
            height, width = canvas_cv.shape[:2]
            print(f"Trying to create video writer with dimensions: {width}x{height}")

            # Try different codecs until one works
            for codec in codecs_to_try:
                try:
                    test_writer = cv2.VideoWriter(output_path, codec, fps, (width, height))
                    if test_writer.isOpened():
                        video_writer = test_writer
                        successful_codec = codec
                        print(f"Successfully initialized video writer with codec: {codec}")
                        break
                    else:
                        test_writer.release()
                except Exception as e:
                    print(f"Failed to initialize with codec {codec}: {e}")
                    continue

            if video_writer is None:
                print("Failed to initialize video writer with any codec, falling back to default")
                video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))  # type: ignore

        # Write frame to video multiple times for longer duration (3 seconds per frame)
        frame_duration_multiplier = 3  # Write each frame 3 times at 1fps = 3 seconds per frame
        for _ in range(frame_duration_multiplier):
            if video_writer and video_writer.isOpened():
                video_writer.write(canvas_cv)
            else:
                print(f"Warning: Video writer not properly initialized for frame {i + 1}")

        print(f"Wrote frame {i + 1}/{num_frames} to video (dimensions: {canvas_cv.shape[:2]})")

        plt.close(fig)
        buf.close()

    # Release video writer
    if video_writer:
        video_writer.release()

        # Verify the video was created successfully
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"Multi-model video saved to: {output_path}")
            print(f"Video file size: {file_size:,} bytes")
            print(f"Expected duration: {num_frames * frame_duration_multiplier / fps:.1f} seconds")
            print(f"Codec used: {successful_codec if successful_codec else 'default mp4v'}")
        else:
            print(f"ERROR: Video file was not created at {output_path}")
    else:
        print("ERROR: Failed to create video writer")


def create_combined_plot(
    all_model_scores: dict, plot_path: str, task_description: str, model_colors: dict, shuffle: bool
):
    """Create a static plot combining all models' scores."""
    plt.figure(figsize=(12, 6))

    # Plot each model's scores
    for model_name, scores in all_model_scores.items():
        color = model_colors.get(model_name, "gray")
        plt.plot(scores, color=color, linewidth=2, marker="o", label=model_name, markersize=4)

    plt.title(f"Task Completion Percentage Comparison\nTask: {task_description}")
    plt.xlabel("Frame Number (Original Order)" if shuffle else "Frame Number")
    plt.ylabel("Task Completion Percentage (%)")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.legend()

    # Save plot
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Combined plot saved to: {plot_path}")


def execute_analysis(
    video_path: str,
    task_description: str = "Moving the chess piece from the red to the blue circle",
    model: Literal["gpt", "gemini-pro", "gemini-flash"] = "gpt",
    frame_interval: int = 30,
    shuffle: bool = True,
) -> tuple[list[float], list[np.ndarray], list[int] | None, list[float], float]:
    """Execute the frame analysis and return results."""
    print("Starting frame analysis...")

    # Extract frames first for video creation
    frames = extract_every_nth_frame(video_path, frame_interval)
    print(f"Extracted {len(frames)} frames (every {frame_interval}th frame)")

    # Analyze with selected model
    values, order_mapping, api_times, estimated_cost = analyze_frames_with_gpt(
        video_path, task_description, model, shuffle=shuffle, frame_interval=frame_interval
    )

    # If shuffled, reorder frames to match the analysis order
    if shuffle and order_mapping:
        frames = [frames[i] for i in order_mapping]

    return values, frames, order_mapping, api_times, estimated_cost


def print_timing_analysis(api_times: list[float], model: str, estimated_cost: float):
    """Print analysis of API request timing and cost."""
    if not api_times:
        print("No API timing data available")
        return

    avg_time = np.mean(api_times)
    first_3_avg = np.mean(api_times[:3]) if len(api_times) >= 3 else np.mean(api_times)
    last_3_avg = np.mean(api_times[-3:]) if len(api_times) >= 3 else np.mean(api_times)

    print(f"\n{'=' * 50}")
    print(f"API REQUEST TIMING & COST ANALYSIS - {model.upper()}")
    print(f"{'=' * 50}")
    print(f"Total requests: {len(api_times)}")
    print(f"Average request time: {avg_time:.2f}s")
    print(f"First 3 requests average: {first_3_avg:.2f}s")
    print(f"Last 3 requests average: {last_3_avg:.2f}s")
    print(f"Time difference (last - first): {last_3_avg - first_3_avg:.2f}s")
    print(f"Fastest request: {min(api_times):.2f}s")
    print(f"Slowest request: {max(api_times):.2f}s")
    print(f"Estimated cost: ${estimated_cost:.4f} USD")
    print(f"{'=' * 50}")


def main(
    video_path: str | None = None,
    task_description: str = "Moving the chess piece from the red to the blue circle",
    models: list[str] | None = None,
    frame_interval: int = 30,
    shuffle: bool = True,
    fps: float = 4.0,
):
    """Main execution function."""
    if models is None:
        models = ["gpt"]
    # Handle "all" models case
    if "all" in models:
        models = ["gpt", "gemini-pro", "gemini-flash"]

    # Use default video if none provided
    if video_path is None:
        video_path = context_video_1_path
        print(f"Using default video: {video_path}")
    else:
        print(f"Using video: {video_path}")

    print(f"Using models: {', '.join(models)}")

    # Store results for all models
    all_results = {}
    total_cost = 0.0

    # Execute analysis for each model
    for model in models:
        print(f"\n{'=' * 60}")
        print(f"ANALYZING WITH MODEL: {model.upper()}")
        print(f"{'=' * 60}")

        try:
            scores, frames, order_mapping, api_times, estimated_cost = execute_analysis(
                video_path,
                task_description,
                model,  # type: ignore
                frame_interval,
                shuffle,
            )

            if not scores:
                print(f"No scores obtained from {model} analysis")
                continue

            # Store results
            all_results[model] = {
                "scores": scores,
                "frames": frames,
                "order_mapping": order_mapping,
                "api_times": api_times,
                "estimated_cost": estimated_cost,
            }

            # Add to total cost
            total_cost += estimated_cost

            # Print timing analysis for this model
            print_timing_analysis(api_times, model, estimated_cost)

            print(f"\nAnalysis complete for {model}!")
            print(f"Number of frames analyzed: {len(scores)}")
            print(f"Score range: {min(scores):.1f}% - {max(scores):.1f}%")
            print(f"Average score: {np.mean(scores):.1f}%")

        except Exception as e:
            print(f"Error analyzing with {model}: {e}")
            continue

    if not all_results:
        print("No successful analyses completed")
        return

    # Print total cost summary
    print(f"\n{'=' * 60}")
    print("TOTAL COST SUMMARY")
    print(f"{'=' * 60}")
    for model, results in all_results.items():
        print(f"{model.upper()}: ${results['estimated_cost']:.4f} USD")
    print(f"{'=' * 60}")
    print(f"TOTAL ESTIMATED COST: ${total_cost:.4f} USD")
    print(f"{'=' * 60}")

    # Create combined visualization
    create_combined_visualization(all_results, task_description, shuffle, video_path, fps)


def create_combined_visualization(all_results: dict, task_description: str, shuffle: bool, video_path: str, fps: float):
    """Create combined video and plot for multiple models."""
    print(f"\nCreating combined visualization for {len(all_results)} models...")

    # Define colors for different models
    model_colors = {"gpt": "blue", "gemini-pro": "green", "gemini-flash": "red"}

    # Get unified frame data from first model
    first_model = list(all_results.keys())[0]
    first_results = all_results[first_model]

    # Extract video name from path (without extension)
    import os

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    shuffle_suffix = "_no-shuffle" if not shuffle else ""

    if len(all_results) == 1:
        # Single model
        model_name = list(all_results.keys())[0]
        video_path = f"artifacts/gvl/{video_name}_{model_name}{shuffle_suffix}_analysis.mp4"
        plot_path = f"artifacts/gvl/{video_name}_{model_name}{shuffle_suffix}_scores.png"
    else:
        # Multiple models
        model_names = "+".join(sorted(all_results.keys()))
        video_path = f"artifacts/gvl/{video_name}_multi-{model_names}{shuffle_suffix}_analysis.mp4"
        plot_path = f"artifacts/gvl/{video_name}_multi-{model_names}{shuffle_suffix}_scores.png"

    # Reorder frames back to original order for video
    if shuffle and first_results["order_mapping"]:
        original_frames = [first_results["frames"][0]] * len(first_results["frames"])
        for shuffled_idx, original_idx in enumerate(first_results["order_mapping"]):
            original_frames[original_idx] = first_results["frames"][shuffled_idx]
        video_frames = original_frames
    else:
        video_frames = first_results["frames"]

    # Prepare all model scores in original order
    all_model_scores = {}
    for model_name, results in all_results.items():
        if shuffle and results["order_mapping"]:
            # Reorder scores back to original frame order
            original_scores = [0.0] * len(results["scores"])
            for shuffled_idx, original_idx in enumerate(results["order_mapping"]):
                original_scores[original_idx] = results["scores"][shuffled_idx]
            all_model_scores[model_name] = original_scores
        else:
            all_model_scores[model_name] = results["scores"]

    # Create visualizations
    create_multimodel_video_with_scores(video_frames, all_model_scores, video_path, model_colors, fps)
    create_combined_plot(all_model_scores, plot_path, task_description, model_colors, shuffle)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GVL - General Vision Language Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                          # Basic usage with defaults
  %(prog)s --video data/my_video.mp4 --model gemini-flash --interval 15
  %(prog)s --model gpt gemini-flash --interval 15   # Multiple models comparison
  %(prog)s --model all --interval 30                # All models comparison
  %(prog)s --task "Moving knight to checkmate" --model gpt gemini-pro --no-shuffle

API Keys Required:
  OPENAI_API_KEY     - For GPT models
  GEMINI_API_KEY     - For Gemini models
        """,
    )

    parser.add_argument(
        "--video", "-v", type=str, default=None, help="Path to video file (default: uses built-in context video)"
    )

    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default="Moving the chess piece from the red to the blue circle",
        help="Task description for analysis",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        nargs="+",
        choices=["gpt", "gemini-pro", "gemini-flash", "all"],
        default=["gpt"],
        help="AI model(s) to use for analysis. Can specify multiple models or 'all' for all models (default: gpt)",
    )

    parser.add_argument(
        "--interval", "-i", type=int, default=30, help="Frame sampling interval - analyze every Nth frame (default: 30)"
    )

    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle frames (analyze in original order)")

    parser.add_argument("--fps", type=float, default=4.0, help="Frames per second for output video (default: 4.0)")

    parser.add_argument("--list-videos", action="store_true", help="List available built-in videos and exit")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_videos:
        print("Available built-in videos:")
        print(f"  context_video_1: {context_video_1_path}")
        print(f"  context_video_2: {context_video_2_path}")
        print(f"  arm_video_1: {arm_video_1_path}")
        print(f"  arm_video_2: {arm_video_2_path}")
        sys.exit(0)

    try:
        main(
            video_path=args.video,
            task_description=args.task,
            models=args.model,  # type: ignore
            frame_interval=args.interval,
            shuffle=not args.no_shuffle,
            fps=args.fps,
        )
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
