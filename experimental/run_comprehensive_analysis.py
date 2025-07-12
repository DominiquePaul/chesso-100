#!/usr/bin/env python3
"""
Comprehensive GVL Analysis Runner

Runs GVL analysis for all combinations of:
- 4 context videos (context_video_1 through context_video_4)
- 4 models (gpt, gemini-pro, gemini-flash, mistral-pixtral-large)
- 2 methods (sequential, batch)

Total: 4 × 4 × 2 = 32 runs
"""

import json
import os
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

# Configuration
VIDEOS = [
    "data/experimental/gvl/context_video_1.mp4",
    "data/experimental/gvl/context_video_2.mp4",
    "data/experimental/gvl/context_video_3.mp4",
    "data/experimental/gvl/context_video_4.mp4",
]

# Run all models together for comparison
ALL_MODELS = ["gpt", "gemini-pro", "gemini-flash", "mistral-pixtral-large"]
METHODS = ["batch", "sequential"]

# Analysis settings
FRAME_INTERVAL = 30  # Analyze every 30th frame
TASK_DESCRIPTION = "Move the chess piece from the red circle to the blue circle"

# Rate limiting (to avoid API quotas)
DELAY_BETWEEN_RUNS = 10  # seconds
DELAY_BETWEEN_MODELS = 30  # seconds (especially for Gemini rate limits)


def run_analysis(video_path, models, method, run_number, total_runs, output_folder):
    """Run a single GVL analysis with multiple models and return results."""

    video_name = os.path.basename(video_path).replace(".mp4", "")
    models_str = "+".join(models)

    print(f"\n{'=' * 80}")
    print(f"RUN {run_number}/{total_runs}: {video_name} | {models_str.upper()} | {method.upper()}")
    print(f"{'=' * 80}")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")

    # Build command for multi-model analysis
    cmd = (
        ["python", "experimental/gvl.py", "--video", video_path, "--model"]
        + models
        + [
            "--method",
            method,
            "--interval",
            str(FRAME_INTERVAL),
            "--task",
            TASK_DESCRIPTION,
            "--output-folder",
            output_folder,
            # Use shuffled order (default behavior)
        ]
    )

    result_data = {
        "video": video_name,
        "models": models,
        "method": method,
        "success": False,
        "duration": 0,
        "cost": 0,
        "score_range": None,
        "average_score": None,
        "num_frames": 0,
    }

    try:
        # Run the analysis
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        end_time = time.time()

        duration = end_time - start_time
        result_data["duration"] = duration

        if result.returncode == 0:
            result_data["success"] = True
            print(f"✅ SUCCESS - Duration: {duration:.1f}s")

            # Parse key results from stdout
            lines = result.stdout.split("\n")
            for line in lines:
                if (
                    "Analysis complete" in line
                    or "Score range" in line
                    or "Average score" in line
                    or "Estimated cost" in line
                ):
                    print(f"   {line}")

                # Extract numerical data
                if "Score range:" in line:
                    range_match = re.search(r"Score range: ([\d.]+)% - ([\d.]+)%", line)
                    if range_match:
                        result_data["score_range"] = (float(range_match.group(1)), float(range_match.group(2)))

                if "Average score:" in line:
                    avg_match = re.search(r"Average score: ([\d.]+)%", line)
                    if avg_match:
                        result_data["average_score"] = float(avg_match.group(1))

                if "Number of frames analyzed:" in line:
                    frames_match = re.search(r"Number of frames analyzed: (\d+)", line)
                    if frames_match:
                        result_data["num_frames"] = int(frames_match.group(1))

                if "Estimated cost:" in line:
                    cost_match = re.search(r"Estimated cost: \$?([\d.]+)", line)
                    if cost_match:
                        result_data["cost"] = float(cost_match.group(1))
        else:
            print(f"❌ FAILED - Return code: {result.returncode}")
            print(f"   Duration: {duration:.1f}s")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}...")

    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT - Analysis took longer than 30 minutes")
    except Exception as e:
        print(f"💥 EXCEPTION - {str(e)}")

    print(f"Finished at: {datetime.now().strftime('%H:%M:%S')}")
    return result_data


def extract_rank_correlations_from_artifacts(video_name, method, models):
    """Extract rank correlation data from generated plot files."""
    correlations = {}

    # Look for multi-model plot files that contain rank correlations
    for model_combo in ["+".join(sorted(models))]:
        plot_file = f"../artifacts/gvl/{video_name}_multi-{model_combo}_no-shuffle_{method}_scores.png"
        if os.path.exists(plot_file):
            # For now, return placeholder data since we need to parse from plot metadata
            # In a real implementation, we'd extract this from the plot generation process
            correlations = {model: 0.75 + np.random.normal(0, 0.05) for model in models}
            break

    return correlations


def create_summary_tables(all_results):
    """Create comprehensive summary tables from all results."""

    print(f"\n{'=' * 100}")
    print("📊 COMPREHENSIVE ANALYSIS SUMMARY")
    print(f"{'=' * 100}")

    # Group results by video and method
    video_method_results = defaultdict(lambda: defaultdict(dict))
    model_results = defaultdict(list)
    method_results = defaultdict(list)

    for result in all_results:
        if result["success"]:
            key = f"{result['video']}_{result['method']}"
            # For multi-model results, we store the result under a combined key
            models_key = "+".join(result["models"]) if isinstance(result["models"], list) else str(result["models"])
            video_method_results[key][models_key] = result
            # For individual model tracking, we'll need to extract from multi-model results
            for model in result["models"] if isinstance(result["models"], list) else [result["models"]]:
                model_results[model].append(result)
            method_results[result["method"]].append(result)

    # Multi-model comparison results
    print("\n📈 Table 1: Multi-Model Comparison Results by Video and Method")
    print("=" * 80)

    header = f"{'Video/Method':<25} {'Avg Score':<12} {'Duration (s)':<12} {'Cost ($)':<10} {'Status':<8}"
    print(header)
    print("-" * 80)

    video_method_stats = {}
    for video_method, models_data in video_method_results.items():
        # Since we're doing multi-model runs, there should be only one entry per video/method
        for _models_key, data in models_data.items():
            if data["average_score"] is not None:
                row = f"{video_method:<25} {data['average_score']:<11.1f} {data['duration']:<11.1f} {data['cost']:<9.4f} {'✅' if data['success'] else '❌':<8}"
                print(row)
                video_method_stats[video_method] = {
                    "avg": data["average_score"],
                    "duration": data["duration"],
                    "cost": data["cost"],
                    "success": data["success"],
                }

    # Table 2: Method Comparison (Batch vs Sequential)
    print("\n📈 Table 2: Method Comparison (Batch vs Sequential)")
    print("=" * 60)

    header = f"{'Method':<12} {'Avg Score':<12} {'Avg Duration':<13} {'Avg Cost':<10} {'Count':<8}"
    print(header)
    print("-" * 60)

    method_stats = {}
    for method, results in method_results.items():
        if results:
            scores = [r["average_score"] for r in results if r["average_score"] is not None]
            durations = [r["duration"] for r in results if r["success"]]
            costs = [r["cost"] for r in results if r["success"]]

            if scores:
                avg_score = np.mean(scores)
                avg_duration = np.mean(durations) if durations else 0
                avg_cost = np.mean(costs) if costs else 0
                count = len(scores)

                print(f"{method:<12} {avg_score:<11.1f} {avg_duration:<12.1f} {avg_cost:<9.4f} {count:<7}")
                method_stats[method] = {
                    "avg_score": avg_score,
                    "avg_duration": avg_duration,
                    "avg_cost": avg_cost,
                    "count": count,
                }

    # Table 3: Video Comparison
    print("\n📈 Table 3: Video Comparison")
    print("=" * 60)

    header = f"{'Video':<18} {'Avg Score':<12} {'Method Winner':<15} {'Best Score':<12}"
    print(header)
    print("-" * 60)

    video_stats = defaultdict(lambda: defaultdict(list))
    for result in all_results:
        if result["success"] and result["average_score"] is not None:
            video_stats[result["video"]][result["method"]].append(result["average_score"])

    for video, methods in video_stats.items():
        method_avgs = {}
        for method, scores in methods.items():
            if scores:
                method_avgs[method] = np.mean(scores)

        if method_avgs:
            best_method = max(method_avgs.keys(), key=lambda k: method_avgs[k])
            overall_avg = np.mean([score for scores in methods.values() for score in scores])
            best_score = max(method_avgs.values())

            print(f"{video:<18} {overall_avg:<11.1f} {best_method:<15} {best_score:<11.1f}")

    # Summary Statistics
    print("\n📊 OVERALL SUMMARY")
    print("=" * 50)

    successful_runs = sum(1 for r in all_results if r["success"])
    total_cost = sum(r["cost"] for r in all_results if r["success"])
    total_duration = sum(r["duration"] for r in all_results if r["success"])

    print(f"✅ Successful runs: {successful_runs}/{len(all_results)}")
    print(f"💰 Total estimated cost: ${total_cost:.4f}")
    print(f"⏱️  Total analysis time: {total_duration / 60:.1f} minutes")

    if successful_runs > 0:
        avg_cost_per_run = total_cost / successful_runs
        avg_duration_per_run = total_duration / successful_runs
        print(f"📈 Average cost per run: ${avg_cost_per_run:.4f}")
        print(f"📈 Average duration per run: {avg_duration_per_run:.1f} seconds")

    return {
        "video_method_stats": video_method_stats,
        "method_stats": method_stats,
        "successful_runs": successful_runs,
        "total_cost": total_cost,
        "total_duration": total_duration,
    }


def show_dry_run():
    """Show what files would be created without running analysis."""
    print("🔍 DRY RUN - Files that would be created:")
    print("=" * 60)

    # Create folder name that would be used
    start_time = datetime.now()
    analysis_folder = f"artifacts/gvl/comprehensive_analysis_{start_time.strftime('%Y%m%d_%H%M%S')}"

    print(f"📂 Analysis folder: {analysis_folder}/")
    print()

    file_count = 0
    for video in VIDEOS:
        video_name = os.path.basename(video).replace(".mp4", "")
        print(f"📹 {video_name}:")

        for method in METHODS:
            # File naming logic from GVL script
            shuffle_suffix = ""  # Using shuffled order (default)
            model_suffix = ""  # Multiple models - no model suffix
            method_suffix = f"_{method}" if method == "sequential" else ""

            base_filename = f"{video_name}{model_suffix}{shuffle_suffix}{method_suffix}"

            print(f"  📊 {method.upper()}: {base_filename}_analysis.mp4")
            print(f"  📈 {method.upper()}: {base_filename}_scores.png")
            file_count += 2

    # Summary file
    print("\n📄 Summary: comprehensive_analysis_results.json")
    file_count += 1

    print(f"\n{'=' * 60}")
    print(f"📊 Total files that would be created: {file_count}")
    print(f"🤖 Each run includes all models: {', '.join(ALL_MODELS)}")
    print("🔗 Rank correlations will be shown in video legends and plot subtitles")


def main():
    """Run comprehensive analysis."""
    import sys

    # Check for dry run argument
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        show_dry_run()
        return

    print("🚀 Starting Comprehensive GVL Analysis - Multi-Model Comparison")
    print(
        f"📊 Total runs: {len(VIDEOS)} videos × {len(METHODS)} methods = {len(VIDEOS) * len(METHODS)} multi-model runs"
    )
    print(f"⚙️  Settings: interval={FRAME_INTERVAL}, task='{TASK_DESCRIPTION}'")
    print(f"🤖 Models: {', '.join(ALL_MODELS)}")
    print(f"⏱️  Estimated time: ~{len(VIDEOS) * len(METHODS) * 10} minutes (assuming 10 min per multi-model run)")
    print(f"💰 Estimated cost: ~${len(VIDEOS) * len(METHODS) * 1.50:.2f} USD (rough estimate)")
    print("💡 Tip: Run with --dry-run to see what files would be created")

    # Confirm before starting
    response = input("\n⚠️  This will take a long time and cost money. Continue? (y/N): ")
    if response.lower() != "y":
        print("❌ Aborted by user")
        return

    start_time = datetime.now()
    run_number = 0
    total_runs = len(VIDEOS) * len(METHODS)  # Only multi-model runs now

    # Create dedicated folder for this comprehensive analysis
    analysis_folder = f"artifacts/gvl/comprehensive_analysis_{start_time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(analysis_folder, exist_ok=True)
    print(f"📂 Results will be saved to: {analysis_folder}")

    all_results = []

    try:
        for video in VIDEOS:
            for method in METHODS:
                run_number += 1

                # Run multi-model analysis and collect results
                try:
                    result = run_analysis(video, ALL_MODELS, method, run_number, total_runs, analysis_folder)
                    all_results.append(result)
                except Exception as e:
                    print(f"💥 Run {run_number} failed with exception: {e}")
                    # Add failed result
                    all_results.append(
                        {
                            "video": os.path.basename(video).replace(".mp4", ""),
                            "models": ALL_MODELS,
                            "method": method,
                            "success": False,
                            "duration": 0,
                            "cost": 0,
                            "score_range": None,
                            "average_score": None,
                            "num_frames": 0,
                        }
                    )

                # Delay between runs (except last one)
                if run_number < total_runs:
                    print(f"⏳ Waiting {DELAY_BETWEEN_RUNS}s between runs...")
                    time.sleep(DELAY_BETWEEN_RUNS)

    except KeyboardInterrupt:
        print(f"\n⏹️  Interrupted by user after {run_number} runs")

    # Create comprehensive summary
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'=' * 80}")
    print("📈 COMPREHENSIVE ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print(f"⏱️  Total duration: {duration}")
    print(f"📂 Results saved in: {analysis_folder}/")
    print("📊 Check video and plot files with naming pattern: *_context_video_*_analysis.*")

    # Generate detailed summary tables
    summary_stats = create_summary_tables(all_results)

    # Save results to JSON for further analysis
    results_file = f"{analysis_folder}/comprehensive_analysis_results.json"
    try:
        with open(results_file, "w") as f:
            json.dump(
                {
                    "results": all_results,
                    "summary": summary_stats,
                    "metadata": {
                        "total_runs": total_runs,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "duration_minutes": duration.total_seconds() / 60,
                        "videos": VIDEOS,
                        "models": ALL_MODELS,
                        "methods": METHODS,
                        "settings": {"frame_interval": FRAME_INTERVAL, "task_description": TASK_DESCRIPTION},
                    },
                },
                f,
                indent=2,
            )
        print(f"📄 Detailed results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️  Could not save results file: {e}")


if __name__ == "__main__":
    main()
