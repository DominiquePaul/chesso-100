# Chesso-100 - A robot that kicks your ass in chess 🤖🍑
> [!WARNING]
> **🚧 Work in Progress**
>
> This repository is actively under development. While the computer vision components are fully functional, the robotic arm integration is still being implemented. Expect frequent updates and potential breaking changes.


> **🤖 Project Overview**
>
> This project combines computer vision and robotics to create an autonomous chess-playing robot using the SO-100 robotic arm. The system uses YOLO-based models for chess piece detection and board analysis, integrates with the Stockfish chess engine for move prediction, and aims to train robotic policies for physical chess piece manipulation. Currently featuring robust computer vision capabilities with ongoing development of the robotic control system.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗-Models_Available-yellow.svg)](https://huggingface.co/dopaul)
[![YOLO](https://img.shields.io/badge/YOLO-v11-red.svg)](https://github.com/ultralytics/ultralytics)

</div>

## ✨ Key Features

🎯 **State-of-the-art Computer Vision**
- Custom YOLO models for chess piece detection with 95%+ accuracy
- Real-time board analysis with pixel-perfect square mapping
- Robust lighting and angle adaptation

🧠 **Smart Chess Engine Integration**
- Stockfish integration with adjustable difficulty levels
- Real-time move prediction and game analysis
- Support for different camera perspectives

🤖 **Robotic Arm Control** *(In Development)*
- SO-100 robotic arm integration via LeRobot framework
- Action Chunking Transformer (ACT) for smooth movements
- Autonomous chess piece manipulation

🚀 **Ready-to-Use Models**
- Pre-trained models available on Hugging Face Hub
- One-command setup and inference
- Comprehensive dataset collection and training scripts

## 🎮 Try It Yourself

**Quick chess analysis in 30 seconds:**

```bash
# Install and analyze any chess image
pip install uv && uv sync
python scripts/analyze_chess_board.py --image your_chess_photo.jpg --computer-playing-as white
```

**What you'll get:**
- 📸 Piece detection with bounding boxes
- 🎯 Board corner detection and grid mapping
- 🏹 AI move predictions with visual arrows
- ♟️ Traditional chess diagram output

**Live chess analysis streaming:**

```bash
# Stream live analysis from your webcam
python scripts/stream_chess_analysis.py --camera 0 --computer-playing-as white

# With recording and custom settings
python scripts/stream_chess_analysis.py --camera 0 --computer-playing-as white --record --fps 2 --verbose
```

**Live streaming features:**
- 🎥 Real-time webcam chess analysis
- 📹 Optional recording to video files
- 🎯 Live move predictions with visual overlays
- ⚡ Adjustable analysis frame rate
- 🎮 Interactive controls (press 'q' to quit, 'r' to force re-analysis)

## Roadmap
### Computer Vision ✅
- [x] **Chess piece detection** - Reliable identification and classification of all chess pieces
- [x] **Board detection** - Robust chessboard corner detection in various lighting conditions
- [x] **Integrated analysis** - Combined board + piece detection with position mapping
- [x] **Chess engine integration** - Board position analysis with move prediction using Stockfish
- [ ] Collect more data for pieces and segmentation (~100 samples each)
- [ ] Final retrain of model

### Robotic Arm 🚧
- [ ] **Data collection interface** - Create user-friendly script to record chess move demonstrations
- [ ] **Movement dataset** - Collect 250+ samples of piece movements with board coordinates
- [ ] **Policy training** - Train ACT (Action Chunking Transformer) model on movement data
- [ ] **Game control loop** - Implement autonomous chess playing system
  - [ ] **Move detection** - Detect when opponent has made their move
  - [ ] **Physical execution** - Execute predicted moves with robotic arm
- [ ] **Commentary system** - Optional AI voice-over for game narration


# Setup

```
pip install uv
uv sync
```

## Get datasets

### Quick Start (Recommended)

```bash
# Download both detection and segmentation datasets
python scripts/download_datasets.py --both

# Download only detection datasets (chess pieces)
python scripts/download_datasets.py --detection

# Download only segmentation datasets (chess board)
python scripts/download_datasets.py --segmentation
```

### 📤 Upload Datasets to HuggingFace

```bash
# Upload both detection and segmentation datasets to HuggingFace Hub
python scripts/upload_datasets_hf.py

# Set your HuggingFace username in .env file:
# HF_USERNAME=yourusername
```

**Created repositories:**
- `chess-pieces-merged` - Combined detection dataset
- `chess-pieces-dominique`, `chess-pieces-roboflow` - Individual detection datasets
- `chess-board-segmentation` - Segmentation dataset with polygon annotations

### 📖 Detailed Instructions

For comprehensive dataset options, API setup, troubleshooting, and advanced usage, see:

**[📋 Complete Dataset Guide →](src/data_prep/README.md)**

The detailed guide covers:
- Multiple download methods (Hugging Face, Roboflow, Kaggle)
- Chess piece detection datasets
- Chessboard corner detection datasets
- API key setup and troubleshooting
- Dataset recreation from source

## 🔗 Resources & Models

<details>
<summary><b>🤗 Pre-trained Models & Datasets</b> (Click to expand)</summary>

### 🎯 Ready-to-Use Models
- **[Chess Piece Detection](https://huggingface.co/dopaul/chess_piece_detection)** - YOLO detection model for identifying and classifying chess pieces
- **[Chess Board Segmentation](https://huggingface.co/dopaul/chess_board_segmentation)** - YOLO segmentation model for precise board boundary detection

### 📊 Training Datasets
- **[Merged Dataset (Recommended)](https://huggingface.co/datasets/dopaul/chess-pieces-merged)** - Combined detection dataset for training
- **[Dominique Dataset](https://huggingface.co/datasets/dopaul/chess-pieces-dominique)** - Individual detection dataset from Roboflow
- **[Roboflow Dataset](https://huggingface.co/datasets/dopaul/chess-pieces-roboflow)** - Processed detection dataset from Kaggle
- **[Chess Board Segmentation](https://huggingface.co/datasets/dopaul/chess-board-segmentation)** - Polygon segmentation dataset for board detection

### 🌐 Original Data Sources
- [Kaggle Chess Pieces Dataset](https://www.kaggle.com/datasets/imtkaggleteam/chess-pieces-detection-image-dataset)
- [Roboflow Chess Pieces Detection](https://universe.roboflow.com/gustoguardian/chess-piece-detection-bltvi/dataset/10)
- [Roboflow Chessboard Corners](https://universe.roboflow.com/gustoguardian/chess-board-box/dataset/3)
- [Roboflow Chessboard Segmentation](https://universe.roboflow.com/gustoguardian/chess-board-i0ptl/dataset/4)

**🎯 Hugging Face Profile**: https://huggingface.co/dopaul

</details>

## Chess Board Analysis with Move Prediction

### 🚀 Quick Start

**Prerequisites:**
1. Install Stockfish chess engine:
   ```bash
   # macOS
   brew install stockfish

   # Ubuntu/Debian
   sudo apt install stockfish

   # Windows: Download from https://stockfishchess.org/download/
   ```

2. Install Python dependencies:
   ```bash
   pip install uv
   uv sync
   ```

### 🎯 Basic Usage

**Analyze a chess board image:**
```bash
python scripts/analyze_chess_board.py --image path/to/chess_image.jpg
```

**With move prediction (Stockfish engine):**
```bash
# Computer playing as white
python scripts/analyze_chess_board.py --image chess.jpg --computer-playing-as white

# Computer playing as black
python scripts/analyze_chess_board.py --image chess.jpg --computer-playing-as black
```

### 🎮 Difficulty Control

**Stockfish skill levels (0-20, where 20 is strongest):**
```bash
# Beginner level
python scripts/analyze_chess_board.py --image chess.jpg --computer-playing-as white --stockfish-skill 5

# Intermediate level
python scripts/analyze_chess_board.py --image chess.jpg --computer-playing-as white --stockfish-skill 10

# Expert level
python scripts/analyze_chess_board.py --image chess.jpg --computer-playing-as white --stockfish-skill 15

# Grandmaster level (default)
python scripts/analyze_chess_board.py --image chess.jpg --computer-playing-as white --stockfish-skill 20
```

**Adjust thinking time:**
```bash
# Quick analysis (0.5 seconds)
python scripts/analyze_chess_board.py --image chess.jpg --computer-playing-as white --engine-time 0.5

# Deep analysis (3 seconds)
python scripts/analyze_chess_board.py --image chess.jpg --computer-playing-as white --engine-time 3.0
```

### 📐 Camera Perspectives

**When your camera is positioned differently:**
```bash
# Camera positioned with white pieces at the top
python scripts/analyze_chess_board.py --image chess.jpg --white-playing-from t

# Camera positioned with white pieces at the left
python scripts/analyze_chess_board.py --image chess.jpg --white-playing-from l

# Camera positioned with white pieces at the right
python scripts/analyze_chess_board.py --image chess.jpg --white-playing-from r

# Default: white pieces at the bottom
python scripts/analyze_chess_board.py --image chess.jpg --white-playing-from b
```

### 🎨 Output Features

The script automatically generates:
- **Move arrows** showing the best move (yellow for white, orange for black)
- **Piece detection** with bounding boxes and confidence scores
- **Board grid** with square numbers and corner detection
- **Chess diagram** with traditional piece symbols
- **Combined visualization** with all plots together

### 📋 Common Examples

```bash
# Complete analysis with beginner AI
python scripts/analyze_chess_board.py \
    --image data/eval_images/chess_4.jpeg \
    --computer-playing-as white \
    --stockfish-skill 8 \
    --engine-time 1.0 \
    --verbose

# Quick robot analysis (fast, medium difficulty)
python scripts/analyze_chess_board.py \
    --image chess.jpg \
    --computer-playing-as black \
    --stockfish-skill 12 \
    --engine-time 0.3

# Analysis without move prediction (just detect pieces)
python scripts/analyze_chess_board.py --image chess.jpg

# Use simple engine instead of Stockfish
python scripts/analyze_chess_board.py \
    --image chess.jpg \
    --computer-playing-as white \
    --engine-type simple
```

### 🔧 All Options

```bash
python scripts/analyze_chess_board.py --help
```

**Key parameters:**
- `--computer-playing-as`: Enable move prediction (`white` or `black`)
- `--stockfish-skill`: Difficulty level (0-20, default: 20)
- `--engine-time`: Thinking time in seconds (default: 1.0)
- `--engine-depth`: Search depth (default: 10)
- `--white-playing-from`: Camera perspective (`b`/`t`/`l`/`r`, default: `b`)
- `--conf`: Piece detection confidence threshold (default: 0.5)
- `--verbose`: Show detailed output

### 📊 Output Files

Results are saved to `artifacts/[image_name]/`:
- `*_corners_and_grid.png` - Board detection and grid overlay
- `*_piece_bounding_boxes.png` - Piece detection with boxes
- `*_piece_centers.png` - Piece centers and coordinates
- `*_chess_diagram.png` - Traditional chess board view
- `*_combined_analysis.png` - All visualizations together

### Quickstart for prediction

[tbd]

### Training the models from scratch

Make sure that you've downloaded the data first.

#### Chess Piece Detection

Train a YOLO object detection model to identify and classify chess pieces on board images.

```bash
# Basic training with default settings (YOLO11s COCO pretrained)
python src/chess_piece_detection/train.py --epochs 100

# Choose different COCO-pretrained model size
python src/chess_piece_detection/train.py \
    --pretrained-model yolo11m.pt \
    --epochs 100

# Custom training parameters
python src/chess_piece_detection/train.py \
    --data data/chess_pieces_merged/data.yaml \
    --pretrained-model yolo11s.pt \
    --epochs 100 \
    --batch 16 \
    --imgsz 640

# Complete example with all options
python src/chess_piece_detection/train.py \
    --data data/chess_pieces_merged/data.yaml \
    --pretrained-model yolo11m.pt \
    --models-folder models/chess_piece_detection \
    --name training_v3 \
    --epochs 100 \
    --batch 32 \
    --imgsz 640 \
    --lr 0.001 \
    --patience 15 \
    --save-period 10 \
    --degrees 10.0 \
    --translate 0.1 \
    --scale 0.2 \
    --fliplr 0.5 \
    --mosaic 1.0 \
    --mixup 0.1 \
    --optimizer AdamW \
    --eval-individual \
    --hf-repo-id "username/chess-piece-detector" \
    --verbose

# View all training options
python src/chess_piece_detection/train.py --help

# Available COCO-pretrained models (YOLO11 - latest architecture):
#   - yolo11n.pt (nano, ~2.6M params, fastest)
#   - yolo11s.pt (small, ~9.4M params, fast, recommended)
#   - yolo11m.pt (medium, ~20.1M params, balanced)
#   - yolo11l.pt (large, ~25.3M params, accurate)
#   - yolo11x.pt (extra large, ~56.9M params, most accurate)

# Run inference example
python -m src.chess_piece_detection.inference_example

# Upload your trained model to Hugging Face Hub
python scripts/upload_hf.py \
    --model models/chess_piece_detection/training_v3/weights/best.pt \
    --repo-name yourusername/chess-piece-detector \
    --model-task detection
```

#### Chessboard Detection (Corner Detection)

We first identify the chessboard corners which we then use to identify the borders with in a second step using classical CV or perspective transformation.

```bash
# Download corner detection dataset
export ROBOFLOW_API_KEY=your_api_key_here
python src/chess_board_detection/download_data.py

# Train corner detection model with default settings
python src/chess_board_detection/yolo/train.py

# Train with custom parameters and HuggingFace upload
python src/chess_board_detection/yolo/train.py \
    --data data/chessboard_corners/chess-board-box-3/data.yaml \
    --epochs 100 \
    --batch 32 \
    --hf-repo-id username/chessboard-detector

# View all training options
python src/chess_board_detection/train.py --help

# Test corner detection
python -m src.chess_board_detection.yolo.inference_example
```

#### Chessboard Segmentation (Polygon Detection)

For more precise chessboard boundary detection, we can use segmentation to get exact polygon coordinates.

```bash
# Download segmentation dataset
export ROBOFLOW_API_KEY=your_api_key_here
python src/chess_board_detection/download_data.py \
    --project gustoguardian/chess-board-i0ptl \
    --version 4 \
    --data-dir data/chessboard_segmentation

# Train segmentation model with default settings (YOLO11n-seg)
python src/chess_board_detection/yolo/segmentation/train_segmentation.py \
    --data data/chessboard_segmentation/chess-board-3/data.yaml \
    --epochs 100

# Train with larger model for better accuracy
python src/chess_board_detection/yolo/segmentation/train_segmentation.py \
    --data data/chessboard_segmentation/chess-board-3/data.yaml \
    --pretrained-model yolo11s-seg.pt \
    --epochs 100 \
    --batch 16 \
    --name polygon_segmentation_training

# View all training options
python src/chess_board_detection/yolo/segmentation/train_segmentation.py --help

# Test segmentation model
python src/chess_board_detection/yolo/segmentation/test_segmentation.py \
    --model artifacts/models/chess_board_segmentation/polygon_segmentation_training/weights/best.pt \
    --image path/to/test_image.jpg
```

#### Upload Models to Hugging Face

Once you've trained your models, you can easily upload them to Hugging Face Hub for sharing and deployment using our unified upload script that supports all three model types (piece detection, corner detection, and segmentation).

```bash
# First, login to Hugging Face
huggingface-cli login

# Upload piece detection model
python scripts/upload_hf.py \
    --model models/chess_piece_detection/training_yolo11s/weights/best.pt \
    --repo-name yourusername/chess-piece-detector \
    --model-task detection

# Upload corner detection model
python scripts/upload_hf.py \
    --model models/chess_board_detection/corner_detection_training/weights/best.pt \
    --repo-name yourusername/chess-corner-detector \
    --model-task corner-detection

# Upload segmentation model
python scripts/upload_hf.py \
    --model artifacts/models/chess_board_segmentation/polygon_segmentation_training/weights/best.pt \
    --repo-name yourusername/chess-segmentation \
    --model-task segmentation

# Upload with custom metadata and training logs
python scripts/upload_hf.py \
    --model artifacts/models/chess_board_segmentation/training/weights/best.pt \
    --repo-name yourusername/chess-board-segmentation \
    --model-task segmentation \
    --description "High-accuracy YOLO segmentation model for chess board polygon detection" \
    --tags computer-vision,chess,yolo,segmentation,polygon-detection \
    --license mit \
    --include-training-dir \
    --verbose

# Test upload without actually uploading
python scripts/upload_hf.py \
    --model path/to/model.pt \
    --repo-name yourusername/model-name \
    --model-task detection \
    --dry-run
```

## Usage Examples

### Chess Board Analysis CLI

Analyze chess board images from the command line using the comprehensive analyzer:

```bash
# Basic analysis - detect corners and pieces
python scripts/analyze_chess_board.py --image data/eval_images/chess_4.jpeg

# Get pixel coordinates for specific squares (useful for robotics)
python scripts/analyze_chess_board.py --image chess.jpg --squares e4,d4,a1,h8

# Skip piece detection mode (board detection only)
python scripts/analyze_chess_board.py --image chess.jpg --skip-piece-detection

# Verbose output with detailed analysis
python scripts/analyze_chess_board.py --image chess.jpg --verbose --output results/

# JSON output for programmatic use
python scripts/analyze_chess_board.py --image chess.jpg --json

# Custom models and confidence threshold
python scripts/analyze_chess_board.py --image chess.jpg --conf 0.7 --segmentation-model path/to/model.pt
```

The CLI provides:
- 🎯 **Corner Detection**: Uses segmentation for precise board boundary detection
- ♟️ **Piece Detection**: Identifies and classifies chess pieces (optional)
- 📍 **Square Coordinates**: Get exact pixel coordinates for any chess square
- 🎨 **Visualizations**: Automatic generation of analysis images
- 📊 **Multiple Formats**: Human-readable output or JSON for automation

### Chess Piece Detection
```python
from src.chess_piece_detection.model import ChessModel

# Load model from Hugging Face
model = ChessModel("dopaul/chess_piece_detection")  # Official model
# OR
model = ChessModel("username/chess-piece-detector")  # Your custom model

# Or create new model with pretrained checkpoint (for training/transfer learning)
model = ChessModel(pretrained_checkpoint="yolo11s.pt")  # Uses YOLO11s by default
model = ChessModel(pretrained_checkpoint="yolo11m.pt")  # Use larger model

# Or load your own trained model
model = ChessModel(model_path="models/chess_piece_detection/training_v3/weights/best.pt")

# Predict pieces on an image
results = model.predict("chess_board_image.jpg", conf=0.5)

# Plot evaluation with bounding boxes and labels
model.plot_eval("chess_board_image.jpg", conf=0.25)

# Get detected piece information
for box in results.boxes:
    class_name = results.names[int(box.cls)]
    confidence = float(box.conf)
    coordinates = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
    print(f"Detected {class_name} with confidence {confidence:.2f} at {coordinates}")
```

### Chessboard Corner Detection
```python
from src.chess_board_detection import ChessBoardModel

# Load model from Hugging Face
corner_model = ChessBoardModel.from_huggingface("dopaul/chess-corner-detection-training")

# Or load from local path
corner_model = ChessBoardModel(model_path="models/corner_detection.pt")

# Detect corners (warns if not exactly 4 found)
results, corner_count, is_valid = corner_model.predict_corners("image.jpg")

# Get precise corner coordinates
coordinates, is_valid = corner_model.get_corner_coordinates("image.jpg")

# Visualize with corner outline
corner_model.plot_eval("image.jpg", show_polygon=True)

# Order corners consistently (top-left, top-right, bottom-right, bottom-left)
if is_valid:
    ordered_corners = corner_model.order_corners(coordinates)
```

### Chessboard Segmentation
```python
from src.chess_board_detection.yolo.segmentation.segmentation_model import ChessBoardSegmentationModel
from ultralytics import YOLO

# Load segmentation model from local path
seg_model = ChessBoardSegmentationModel(model_path="artifacts/models/chess_board_segmentation/training/weights/best.pt")

# Or load directly from Hugging Face
yolo_model = YOLO("dopaul/chess_board_segmentation")

# Get precise polygon coordinates
polygon_info, is_valid = seg_model.get_polygon_coordinates("image.jpg")

# Visualize segmentation with polygon outline
seg_model.plot_eval("image.jpg", show_polygon=True, show_mask=True)

# Extract polygon points for perspective transformation
if is_valid:
    coordinates = polygon_info['coordinates']
    print(f"Detected {len(coordinates)} polygon points")
    for i, point in enumerate(coordinates):
        print(f"Point {i}: ({point['x']:.1f}, {point['y']:.1f})")

# Extract corners from segmentation for board alignment
corners = seg_model.extract_corners_from_segmentation("image.jpg", polygon_info)
print(f"Extracted corners: {corners}")
```

## 🔧 Development

This project uses Ruff for linting, Bandit for security, and pre-commit hooks for quality checks.

To set up development environment:
```bash
pip install pre-commit
pre-commit install
```

## 🔧 Troubleshooting

### Common Issues

#### Dataset Download Problems
- **API Key Error**: Follow [Roboflow API key instructions](#-how-to-get-free-roboflow-api-key) above or use manual download
- **Network Issues**: Check internet connection and try different download methods
- **Missing Dependencies**: Run `uv add roboflow huggingface_hub`

#### Training Issues
- **Dataset Not Found**: Verify data path with `--data` flag
- **GPU Memory**: Reduce `--batch` size if running out of memory
- **HuggingFace Upload**: Login with `huggingface-cli login` first

See detailed troubleshooting guide: [src/data_prep/README.md](src/data_prep/README.md#troubleshooting)


# Recording data for the SO-100

Use the script below to collect data. It assumes you are using two cameras so you might want to change this.

```
python -m src.record_lerobot_dataset \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.usbmodem59700741781 \
  --robot.id=doms_follower_arm \
  --robot.cameras="{ context: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}, arm: {type: opencv, index_or_path: 1, width: 1920, height: 1080, fps: 30}}" \
  --teleop.type=so100_leader \
  --teleop.port=/dev/tty.usbmodem59700724381 \
  --teleop.id=doms_leader_arm \
  --dataset.repo_id=dopaul/test_no_video \
  --dataset.num_episodes=50 \
  --dataset.single_task="Move the chess piece from red to blue" \
  --dataset.num_image_writer_processes=8 \
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=1 \
  --dataset.video false \
  --resume true
```

# Train a policy using ACT

**Note**: Download the dataset directly to your HuggingFace cache directory to avoid symlink issues that can prevent the training script from recognizing the dataset properly.

`huggingface-cli download dopaul/100_rooks --repo-type dataset --local-dir ~/.cache/huggingface/lerobot/dopaul/100_rooks`

### Train a policy:

**Careful ⚠️** My laptop crashed because it consumed 60gb of RAM. I trained on a H100 GPU on lightning.ai.

```
python lerobot/lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=dopaul/100_rooks \
    --batch_size 8 \
    --steps 100000 \
    --eval_freq 20000 \
    --log_freq 200 \
    --dataset.video_backend=pyav \
    --save_checkpoint true  \
    --save_freq 20_000 \
    --wandb.enable true \
    --wandb.entity 'dominique-paul' \
    --wandb.project chesso
```

If you want to test the script with a smaller dataset you could use `dopaul/first_movement_test_v5`. It has 3 samples or reduce the batch size.

### Evaluate a policy

```
python -m src.record_lerobot_dataset \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.usbmodem59700741781 \
  --robot.id=doms_follower_arm \
  --robot.cameras="{ context: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}, arm: {type: opencv, index_or_path: 1, width: 1920, height: 1080, fps: 30}}" \
  --dataset.repo_id=dopaul/eval_100_rook_unclipped_act_20k \
  --dataset.num_episodes=25 \
  --dataset.single_task="Move the chess piece from red to blue" \
  --dataset.num_image_writer_processes=8 \
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=2 \
  --teleop.type=so100_leader \
  --teleop.port=/dev/tty.usbmodem59700724381 \
  --teleop.id=doms_leader_arm \
  --policy.path=dopaul/100_rooks_unclipped_act_20k
```
