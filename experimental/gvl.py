#!/usr/bin/env python3
"""
GVL (General Vision Language) Analysis Tool

Analyze video frames with AI models (GPT-4o, Gemini Pro, Gemini Flash) to evaluate task completion.

Examples:
    # Basic usage with default settings (batch mode)
    python gvl.py

    # Sequential mode with conversation history
    python gvl.py --method sequential

    # Batch mode (all frames at once)
    python gvl.py --method batch

    # Single model with custom settings
    python gvl.py --video data/experimental/gvl/arm_video_2.mp4 --model gemini-flash --interval 15

    # Multiple models comparison with sequential method
    python gvl.py --model gpt gemini-flash mistral-pixtral-large --interval 15 --method sequential

    # All models comparison with batch method
    python gvl.py --model all --interval 30 --method batch

    # Chess analysis with multiple models, no shuffling, sequential method
    python gvl.py --task "Moving knight to checkmate position" --model gpt gemini-pro mistral-pixtral-large --no-shuffle --method sequential

    # Quick analysis with all models using batch method
    python gvl.py --model all --interval 60 --task "Picking up the red cup" --method batch

API Keys Required:
    - For GPT: Set OPENAI_API_KEY environment variable
    - For Gemini: Set GEMINI_API_KEY environment variable
    - For Mistral: Set MISTRAL_API_KEY environment variable
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
from scipy.stats import spearmanr

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
    "mistral-pixtral-large": {
        "input_cost_per_1k_tokens": 0.003,  # Pixtral Large input cost
        "output_cost_per_1k_tokens": 0.009,  # Pixtral Large output cost
        "estimated_input_tokens_per_image": 4096,  # Pixtral Large max tokens per image
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


def estimate_api_cost_batch(model: str, num_images: int) -> float:
    """Estimate API cost for batch mode (single request with all images)."""
    if model not in API_COSTS:
        return 0.0

    costs = API_COSTS[model]

    # For batch mode: single request with all images
    text_tokens = 200  # System prompt + user prompt
    image_tokens = costs["estimated_input_tokens_per_image"] * num_images
    total_input_tokens = text_tokens + image_tokens

    # Single response covering all frames
    total_output_tokens = costs["estimated_output_tokens_per_response"] * 2  # Longer response for multiple frames

    # Calculate total cost
    input_cost = (total_input_tokens / 1000) * costs["input_cost_per_1k_tokens"]
    output_cost = (total_output_tokens / 1000) * costs["output_cost_per_1k_tokens"]

    return input_cost + output_cost


def call_model_api_sequential(
    model: Literal["gpt", "gemini-pro", "gemini-flash", "mistral-pixtral-large"],
    messages: list[dict],
    system_prompt: str,
    image_b64: str,
    prompt_text: str,
) -> tuple[str, float]:
    """Call the specified model API for sequential processing with conversation history."""
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

        elif model == "mistral-pixtral-large":
            # Mistral Pixtral Large
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY environment variable not set")

            # Use OpenAI client for Mistral API (compatible interface)
            client = openai.OpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1")

            response = client.chat.completions.create(
                model="pixtral-large-latest",
                messages=messages,  # type: ignore
                max_tokens=150,
            )
            response_content = response.choices[0].message.content or ""

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


def call_model_api_batch(
    model: Literal["gpt", "gemini-pro", "gemini-flash", "mistral-pixtral-large"],
    frames_b64: list[str],
    system_prompt: str,
    prompt_text: str,
) -> tuple[str, float]:
    """Call the specified model API with all frames at once and return response content and timing."""
    start_time = time.time()

    try:
        if model == "gpt":
            # OpenAI GPT-4o - Build message with all frames
            content: list[Any] = [{"type": "text", "text": prompt_text}]

            # Add all frames to the content
            for _i, frame_b64 in enumerate(frames_b64):
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}})

            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]

            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,  # type: ignore
                max_tokens=500,  # Increase for multiple frames
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

            # Convert all base64 frames to PIL Images for Gemini
            content_list: list[str | Image.Image] = [f"{system_prompt}\n\n{prompt_text}"]
            for _i, frame_b64 in enumerate(frames_b64):
                image_data = base64.b64decode(frame_b64)
                image_pil = Image.open(io.BytesIO(image_data))
                content_list.append(image_pil)

            response = gemini_model.generate_content(content_list)
            response_content = response.text

        elif model == "mistral-pixtral-large":
            # Mistral Pixtral Large - Build message with all frames
            content: list[Any] = [{"type": "text", "text": prompt_text}]

            # Add all frames to the content
            for _i, frame_b64 in enumerate(frames_b64):
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}})

            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]

            # Mistral API
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY environment variable not set")

            client = openai.OpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1")

            response = client.chat.completions.create(
                model="pixtral-large-latest",
                messages=messages,  # type: ignore
                max_tokens=500,  # Increase for multiple frames
            )
            response_content = response.choices[0].message.content or ""

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


def analyze_frames_sequential(
    video_path: str,
    task_description: str = "Moving the chess piece from the red to the blue circle",
    model: Literal["gpt", "gemini-pro", "gemini-flash", "mistral-pixtral-large"] = "gpt",
    shuffle: bool = False,
    frame_interval: int = 30,
) -> tuple[list[float], list[int] | None, list[float], float]:
    """Analyze frames from video with AI models for task completion using sequential method with conversation history."""
    # Extract frames
    frames = extract_every_nth_frame(video_path, frame_interval)
    if not frames:
        return [], None, [], 0.0

    # Calculate estimated cost for sequential method
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
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        # Add previous context (images and responses)
        for prev_frame_b64, prev_response in conversation_history:
            prev_content: list[Any] = [
                {"type": "text", "text": "Previous frame:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{prev_frame_b64}"}},
            ]
            messages.append({"role": "user", "content": prev_content})
            messages.append({"role": "assistant", "content": prev_response})

        # Add current frame
        if i == 0:
            frame_text = f"Initial robot scene for task: {task_description}. What is the task completion percentage for this frame?"
        else:
            frame_text = (
                f"Frame {i + 1} for task: {task_description}. What is the task completion percentage for this frame?"
            )

        current_content: list[Any] = [
            {"type": "text", "text": frame_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
        ]
        messages.append({"role": "user", "content": current_content})

        # Make API call with timing
        response_content, api_time = call_model_api_sequential(
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


def analyze_frames_batch(
    video_path: str,
    task_description: str = "Moving the chess piece from the red to the blue circle",
    model: Literal["gpt", "gemini-pro", "gemini-flash", "mistral-pixtral-large"] = "gpt",
    shuffle: bool = False,
    frame_interval: int = 30,
) -> tuple[list[float], list[int] | None, list[float], float]:
    """Analyze frames from video with AI models for task completion - all frames at once."""
    # Extract frames
    frames = extract_every_nth_frame(video_path, frame_interval)
    if not frames:
        return [], None, [], 0.0

    # Calculate estimated cost (single request now)
    estimated_cost = estimate_api_cost_batch(model, len(frames))

    # Create shuffled order if requested
    original_order_mapping = None
    if shuffle:
        indices = list(range(len(frames)))
        shuffled_indices = [0] + random.sample(indices[1:], len(indices) - 1)
        original_order_mapping = shuffled_indices.copy()
        frames = [frames[i] for i in shuffled_indices]

    # Convert all frames to base64
    frames_b64 = [frame_to_base64(frame) for frame in frames]

    # Create the system prompt for batch analysis
    system_prompt = f"""You are an expert roboticist tasked to predict task completion percentages for frames of a robot performing the task: {task_description}.

The task completion percentages are between 0 and 100, where 100 corresponds to full task completion.

Analyze ALL the provided frames and provide task completion percentages. Note that these frames may be in random order, so please analyze each frame independently.

For each frame, format your response exactly as:
Frame 1: Task Completion Percentage: X%
Frame 2: Task Completion Percentage: Y%
Frame 3: Task Completion Percentage: Z%
...and so on for all frames.

Focus on the key visual indicators that show progress toward {task_description}."""

    # Create prompt for all frames
    prompt_text = f"Please analyze these {len(frames)} frames showing a robot performing the task: {task_description}. Provide the task completion percentage for each frame in order."

    print(f"Making single API call with {len(frames)} frames...")

    # Make single API call with all frames
    response_content, api_time = call_model_api_batch(
        model=model, frames_b64=frames_b64, system_prompt=system_prompt, prompt_text=prompt_text
    )

    # Parse percentages from batch response
    values = []
    try:
        if not response_content:
            print("Empty response from API")
            return [0.0] * len(frames), original_order_mapping, [api_time], estimated_cost

        print(f"Raw API response:\n{response_content}\n")

        # Extract percentages for each frame
        frame_pattern = r"Frame\s+(\d+):\s*.*?Task\s+Completion\s+Percentage:\s*(\d+(?:\.\d+)?)%"
        matches = re.findall(frame_pattern, response_content, re.IGNORECASE)

        if matches:
            # Sort by frame number and extract percentages
            frame_percentages = {}
            for frame_num_str, percentage_str in matches:
                frame_num = int(frame_num_str)
                percentage = float(percentage_str)
                frame_percentages[frame_num] = percentage

            # Create ordered list of percentages
            for i in range(1, len(frames) + 1):
                if i in frame_percentages:
                    values.append(frame_percentages[i])
                    print(f"Frame {i}: {frame_percentages[i]}%")
                else:
                    values.append(0.0)
                    print(f"Frame {i}: 0.0% (not found in response)")
        else:
            # Fallback: look for any percentages in order
            fallback_matches = re.findall(r"(\d+(?:\.\d+)?)%", response_content)
            if fallback_matches:
                for i, match in enumerate(fallback_matches[: len(frames)]):
                    percentage = float(match)
                    values.append(percentage)
                    print(f"Frame {i + 1}: {percentage}% (fallback parsing)")

                # Fill remaining frames if needed
                while len(values) < len(frames):
                    values.append(0.0)
                    print(f"Frame {len(values)}: 0.0% (fallback default)")
            else:
                print("Could not parse any percentages from response")
                values = [0.0] * len(frames)

    except Exception as e:
        print(f"Error parsing batch response: {e}")
        print(f"Response: {response_content}")
        values = [0.0] * len(frames)

    return values, original_order_mapping, [api_time], estimated_cost


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

        # Calculate rank correlations for video
        rank_correlations = calculate_rank_correlations(all_model_scores)

        # Plot each model's scores up to current frame
        for model_name, scores in all_model_scores.items():
            current_scores = scores[: i + 1]
            current_indices = list(range(len(current_scores)))
            color = model_colors.get(model_name, "gray")

            # Add rank correlation to label if available and multiple models
            if model_name in rank_correlations and len(all_model_scores) > 1:
                correlation = rank_correlations[model_name]
                label = f"{model_name} ({correlation:.3f})"
            else:
                label = model_name

            ax.plot(current_indices, current_scores, color=color, linewidth=3, label=label)  # Thicker lines
            ax.scatter(current_indices, current_scores, color=color, s=30, zorder=5)  # Larger dots
            if i < len(scores):
                ax.scatter([i], [scores[i]], color=color, s=60, zorder=6)  # Larger current point

        ax.set_xlim(0, num_frames - 1)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Frame", fontsize=10)  # Larger font
        ax.set_ylabel("Completion %", fontsize=10)  # Larger font

        # Update title to include rank correlation explanation
        title = f"Progress (Frame {i + 1})"
        if len(all_model_scores) > 1:
            title += "\n(parentheses show avg rank correlation)"
        ax.set_title(title, fontsize=10)  # Slightly smaller to fit subtitle
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)  # Larger tick labels

        # Add legend if multiple models
        if len(all_model_scores) > 1:
            ax.legend(fontsize=7, loc="upper left")  # Slightly smaller font to fit correlations

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


def calculate_rank_correlations(all_model_scores: dict) -> dict:
    """Calculate rank correlations between models."""
    model_names = list(all_model_scores.keys())
    rank_correlations = {}

    if len(model_names) < 2:
        return rank_correlations

    # Calculate pairwise rank correlations
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i != j:
                scores1 = all_model_scores[model1]
                scores2 = all_model_scores[model2]

                # Calculate Spearman rank correlation
                correlation, p_value = spearmanr(scores1, scores2)
                rank_correlations[f"{model1}_vs_{model2}"] = correlation

    # Calculate average rank correlation for each model (against all others)
    avg_rank_correlations = {}
    for model in model_names:
        correlations = []
        for other_model in model_names:
            if model != other_model:
                key = f"{model}_vs_{other_model}"
                if key in rank_correlations:
                    correlations.append(rank_correlations[key])

        if correlations:
            avg_rank_correlations[model] = np.mean(correlations)
        else:
            avg_rank_correlations[model] = 0.0

    return avg_rank_correlations


def create_combined_plot(
    all_model_scores: dict, plot_path: str, task_description: str, model_colors: dict, shuffle: bool
):
    """Create a static plot combining all models' scores."""
    plt.figure(figsize=(12, 6))

    # Calculate rank correlations
    rank_correlations = calculate_rank_correlations(all_model_scores)

    # Plot each model's scores
    for model_name, scores in all_model_scores.items():
        color = model_colors.get(model_name, "gray")
        # Add rank correlation to label if available
        if model_name in rank_correlations and len(all_model_scores) > 1:
            correlation = rank_correlations[model_name]
            label = f"{model_name} ({correlation:.3f})"
        else:
            label = model_name
        plt.plot(scores, color=color, linewidth=2, marker="o", label=label, markersize=4)

    subtitle = "Task Completion Percentage Comparison\n"
    if len(all_model_scores) > 1:
        subtitle += f"Task: {task_description} (parentheses show average rank correlation)"
    else:
        subtitle += f"Task: {task_description}"

    plt.title(subtitle)
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
    model: Literal["gpt", "gemini-pro", "gemini-flash", "mistral-pixtral-large"] = "gpt",
    frame_interval: int = 30,
    shuffle: bool = True,
    method: Literal["sequential", "batch"] = "batch",
) -> tuple[list[float], list[np.ndarray], list[int] | None, list[float], float]:
    """Execute the frame analysis and return results."""
    print(f"Starting frame analysis using {method.upper()} method...")

    # Extract frames first for video creation
    frames = extract_every_nth_frame(video_path, frame_interval)
    print(f"Extracted {len(frames)} frames (every {frame_interval}th frame)")

    if method == "sequential":
        values, order_mapping, api_times, estimated_cost = analyze_frames_sequential(
            video_path, task_description, model, shuffle=shuffle, frame_interval=frame_interval
        )
    else:  # batch
        values, order_mapping, api_times, estimated_cost = analyze_frames_batch(
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
    method: Literal["sequential", "batch"] = "batch",
    output_folder: str = "artifacts/gvl",
):
    """Main execution function."""
    if models is None:
        models = ["gpt"]
    # Handle "all" models case
    if "all" in models:
        models = ["gpt", "gemini-pro", "gemini-flash", "mistral-pixtral-large"]

    # Use default video if none provided
    if video_path is None:
        video_path = context_video_1_path
        print(f"Using default video: {video_path}")
    else:
        print(f"Using video: {video_path}")

    print(f"Using models: {', '.join(models)}")
    print(f"Using analysis method: {method.upper()}")

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
                method,
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
    create_combined_visualization(all_results, task_description, shuffle, video_path, fps, method, output_folder)


def create_combined_visualization(
    all_results: dict,
    task_description: str,
    shuffle: bool,
    video_path: str,
    fps: float,
    method: str,
    output_folder: str = "artifacts/gvl",
):
    """Create combined video and plot for multiple models."""
    print(f"\nCreating combined visualization for {len(all_results)} models...")

    # Define colors for different models
    model_colors = {"gpt": "blue", "gemini-pro": "green", "gemini-flash": "red", "mistral-pixtral-large": "purple"}

    # Get unified frame data from first model
    first_model = list(all_results.keys())[0]
    first_results = all_results[first_model]

    # Extract video name from path (without extension)
    import os

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    shuffle_suffix = "_no-shuffle" if not shuffle else ""

    # Improved file naming logic
    if len(all_results) == 1:
        # Single model - include model name
        model_name = list(all_results.keys())[0]
        model_suffix = f"_{model_name}"
    else:
        # Multiple models - no model suffix (cleaner default)
        model_suffix = ""

    # Method suffix - only add if sequential
    method_suffix = f"_{method}" if method == "sequential" else ""

    # Construct file paths with custom output folder
    base_filename = f"{video_name}{model_suffix}{shuffle_suffix}{method_suffix}"
    video_output_path = f"{output_folder}/{base_filename}_analysis.mp4"
    plot_output_path = f"{output_folder}/{base_filename}_scores.png"

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
    create_multimodel_video_with_scores(video_frames, all_model_scores, video_output_path, model_colors, fps)
    create_combined_plot(all_model_scores, plot_output_path, task_description, model_colors, shuffle)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GVL - General Vision Language Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                          # Basic usage with defaults (batch mode)
  %(prog)s --video data/my_video.mp4 --model gemini-flash --interval 15
  %(prog)s --model gpt gemini-flash mistral-pixtral-large --interval 15   # Multiple models comparison (batch mode)
  %(prog)s --model all --interval 30                # All models comparison (batch mode)
  %(prog)s --task "Moving knight to checkmate" --model gpt gemini-pro mistral-pixtral-large --no-shuffle --method sequential
  %(prog)s --model all --interval 60 --task "Picking up the red cup" --method sequential
  %(prog)s --method batch --model gpt               # Explicit batch mode (default)
  %(prog)s --method sequential --model gemini-flash # Sequential with conversation history

Method Comparison:
  --method batch      : Faster, cheaper, analyzes all frames in single API call
  --method sequential : Slower, more expensive, builds conversation history for context

API Keys Required:
  OPENAI_API_KEY     - For GPT models
  GEMINI_API_KEY     - For Gemini models
  MISTRAL_API_KEY    - For Mistral models
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
        choices=["gpt", "gemini-pro", "gemini-flash", "mistral-pixtral-large", "all"],
        default=["gpt"],
        help="AI model(s) to use for analysis. Can specify multiple models or 'all' for all models (default: gpt)",
    )

    parser.add_argument(
        "--interval", "-i", type=int, default=30, help="Frame sampling interval - analyze every Nth frame (default: 30)"
    )

    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle frames (analyze in original order)")

    parser.add_argument("--fps", type=float, default=4.0, help="Frames per second for output video (default: 4.0)")

    parser.add_argument(
        "--method",
        type=str,
        choices=["sequential", "batch"],
        default="batch",
        help="Analysis method: 'sequential' uses conversation history (slower, more expensive, better context), 'batch' analyzes all frames at once (faster, cheaper) (default: batch)",
    )

    parser.add_argument("--list-videos", action="store_true", help="List available built-in videos and exit")

    parser.add_argument(
        "--output-folder", type=str, default="artifacts/gvl", help="Output folder for results (default: artifacts/gvl)"
    )

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
            method=args.method,
            output_folder=args.output_folder,
        )
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
