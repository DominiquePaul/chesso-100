#!/usr/bin/env python3
"""
Simple Camera Streaming Script

Stream live video from webcam using WebcamCapture class.

USAGE:
======
# Stream from default camera with default settings (camera 0, 640x480, 30fps)
python scripts/stream_camera.py

# Record frames while streaming (default is that every 25th frame is saved to a new directory in artifacts/)
python scripts/stream_camera.py --record

# Stream from specific camera
python scripts/stream_camera.py --camera 1

# Custom resolution and FPS
python scripts/stream_camera.py --camera 0 --resolution 1280x720 --fps 30

# Record to custom output directory
python scripts/stream_camera.py --record --output-dir ./my_recordings

# Test camera capabilities without streaming
python scripts/stream_camera.py --test-capabilities

# Interactive Camera Controls (while streaming):
# Focus Controls:
#   'a' - Enable autofocus (if supported)
#   'm' - Enable manual focus (if supported)
#   '+' or '=' - Increase manual focus (focus farther)
#   '-' - Decrease manual focus (focus nearer)
#
# Exposure Controls:
#   'e' - Toggle auto exposure (if supported)
#   'E' - Enable manual exposure mode (if supported)
#   'w' - Decrease exposure (reduce light sensitivity)
#   's' - Increase exposure (increase light sensitivity)
#
# Light Controls:
#   'r' - Decrease gain/ISO (reduce light sensitivity)
#   't' - Increase gain/ISO (increase light sensitivity)
#   'z' - Decrease brightness (if supported)
#   'x' - Increase brightness (if supported)
#
# Other Controls:
#   'f' - Toggle overlay display
#   'd' - Show camera diagnostics
#   'q' - Quit
#
# Note: Only controls supported by your camera will be active. Use --test-capabilities to see what's available.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from camera import WebcamCapture


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simple camera streaming script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device index (default: 0)")

    parser.add_argument(
        "--resolution", type=str, default="640x480", help="Camera resolution in WIDTHxHEIGHT format (default: 640x480)"
    )

    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")

    parser.add_argument(
        "--window-name", type=str, default="Camera Stream", help="Window name for display (default: 'Camera Stream')"
    )

    parser.add_argument("--record", action="store_true", help="Record frames during streaming (default: False)")

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for recorded frames. If not specified and --record is set, creates a timestamped directory in artifacts/",
    )

    parser.add_argument(
        "--save-n-frames", type=int, default=25, help="Save every Nth frame when recording (default: 25)"
    )

    parser.add_argument("--disable-autofocus", action="store_true", help="Start with manual focus instead of autofocus")

    parser.add_argument(
        "--manual-focus",
        type=float,
        help="Initial manual focus distance (0.0=near, 1.0=far). Implies --disable-autofocus",
    )

    parser.add_argument("--test-capabilities", action="store_true", help="Test camera capabilities and exit")

    return parser.parse_args()


class CameraCapabilities:
    """Stores information about what camera controls are actually supported."""

    def __init__(self, camera_cap):
        self.autofocus_supported = False
        self.manual_focus_supported = False
        self.auto_exposure_supported = False
        self.manual_exposure_supported = False
        self.gain_supported = False
        self.brightness_supported = False
        self.sharpness_supported = False

        # Value ranges for supported controls
        self.focus_range = (0.0, 1.0)
        self.exposure_range = (-13.0, 1.0)  # Common range for most cameras
        self.gain_range = (0.0, 100.0)
        self.brightness_range = (0.0, 100.0)

        # Test capabilities
        self._test_capabilities(camera_cap)

    def _test_capabilities(self, cap):
        """Test what camera controls are actually supported."""
        if cap is None:
            return

        print("🔍 Testing camera capabilities...")

        # Test autofocus
        try:
            old_af = cap.get(cv2.CAP_PROP_AUTOFOCUS)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            new_af = cap.get(cv2.CAP_PROP_AUTOFOCUS)
            self.autofocus_supported = abs(new_af - 1.0) < 0.1
            cap.set(cv2.CAP_PROP_AUTOFOCUS, old_af)  # restore
            print(f"   Autofocus: {'✅ YES' if self.autofocus_supported else '❌ NO'}")
        except Exception:
            print("   Autofocus: ❌ NO (error)")

        # Test manual focus
        try:
            old_focus = cap.get(cv2.CAP_PROP_FOCUS)
            cap.set(cv2.CAP_PROP_FOCUS, 0.0)
            near_focus = cap.get(cv2.CAP_PROP_FOCUS)
            cap.set(cv2.CAP_PROP_FOCUS, 1.0)
            far_focus = cap.get(cv2.CAP_PROP_FOCUS)
            cap.set(cv2.CAP_PROP_FOCUS, old_focus)  # restore

            self.manual_focus_supported = abs(near_focus - far_focus) > 0.01
            if self.manual_focus_supported:
                self.focus_range = (min(near_focus, far_focus), max(near_focus, far_focus))
            print(f"   Manual focus: {'✅ YES' if self.manual_focus_supported else '❌ NO'}")
        except Exception:
            print("   Manual focus: ❌ NO (error)")

        # Test auto exposure
        try:
            old_ae = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            new_ae = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            self.auto_exposure_supported = abs(new_ae - 1.0) < 0.1
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, old_ae)  # restore
            print(f"   Auto exposure: {'✅ YES' if self.auto_exposure_supported else '❌ NO'}")
        except Exception:
            print("   Auto exposure: ❌ NO (error)")

        # Test manual exposure
        try:
            old_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
            # Try different exposure values to see if camera responds
            test_values = [-10, -5, 0, -1]
            exposure_values = []

            for test_val in test_values:
                cap.set(cv2.CAP_PROP_EXPOSURE, test_val)
                actual_val = cap.get(cv2.CAP_PROP_EXPOSURE)
                exposure_values.append(actual_val)

            cap.set(cv2.CAP_PROP_EXPOSURE, old_exposure)  # restore

            # Check if we got different values
            unique_values = set(exposure_values)
            self.manual_exposure_supported = len(unique_values) > 1

            if self.manual_exposure_supported:
                self.exposure_range = (min(exposure_values), max(exposure_values))
            print(f"   Manual exposure: {'✅ YES' if self.manual_exposure_supported else '❌ NO'}")
        except Exception:
            print("   Manual exposure: ❌ NO (error)")

        # Test gain
        try:
            old_gain = cap.get(cv2.CAP_PROP_GAIN)
            cap.set(cv2.CAP_PROP_GAIN, 0)
            low_gain = cap.get(cv2.CAP_PROP_GAIN)
            cap.set(cv2.CAP_PROP_GAIN, 100)
            high_gain = cap.get(cv2.CAP_PROP_GAIN)
            cap.set(cv2.CAP_PROP_GAIN, old_gain)  # restore

            self.gain_supported = abs(low_gain - high_gain) > 0.01
            if self.gain_supported:
                self.gain_range = (min(low_gain, high_gain), max(low_gain, high_gain))
            print(f"   Gain control: {'✅ YES' if self.gain_supported else '❌ NO'}")
        except Exception:
            print("   Gain control: ❌ NO (error)")

        # Test brightness
        try:
            old_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
            low_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
            high_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, old_brightness)  # restore

            self.brightness_supported = abs(low_brightness - high_brightness) > 0.01
            if self.brightness_supported:
                self.brightness_range = (min(low_brightness, high_brightness), max(low_brightness, high_brightness))
            print(f"   Brightness control: {'✅ YES' if self.brightness_supported else '❌ NO'}")
        except Exception:
            print("   Brightness control: ❌ NO (error)")

        # Test sharpness
        try:
            old_sharpness = cap.get(cv2.CAP_PROP_SHARPNESS)
            self.sharpness_supported = old_sharpness >= 0  # Basic check
            print(f"   Sharpness control: {'✅ YES' if self.sharpness_supported else '❌ NO'}")
        except Exception:
            print("   Sharpness control: ❌ NO (error)")

    def get_available_controls(self):
        """Return a list of available controls for display."""
        controls = []
        if self.autofocus_supported or self.manual_focus_supported:
            controls.append("Focus: a=Auto m=Manual +/-=Adjust")
        if self.auto_exposure_supported or self.manual_exposure_supported:
            controls.append("Exposure: e=Toggle E=Manual w/s=Adjust")
        if self.gain_supported:
            controls.append("Gain: r/t=Adjust")
        if self.brightness_supported:
            controls.append("Brightness: z/x=Adjust")
        return controls


def create_output_directory(output_dir: str | None = None) -> Path:
    """Create and return the output directory for recording."""
    if output_dir:
        output_path = Path(output_dir)
    else:
        # Create timestamped directory in artifacts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        output_path = artifacts_dir / f"camera_recording_{timestamp}"

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def display_stream_with_recording(
    camera: WebcamCapture,
    window_name: str,
    record: bool,
    output_dir: Path | None = None,
    save_n_frames: int = 10,
    capabilities: CameraCapabilities | None = None,
):
    """
    Display the video stream with optional recording capability.

    Args:
        camera: WebcamCapture instance
        window_name: Name of the display window
        record: Whether to record frames
        output_dir: Directory to save frames (if recording)
        save_n_frames: Save every Nth frame
    """
    if not camera.start_streaming():
        return

    frame_count = 0
    saved_count = 0
    last_frame_hash = None
    start_time = time.time() if record else None
    last_status_update = time.time() if record else None

    # Focus control variables
    manual_focus_mode = not camera.enable_autofocus
    manual_focus_value = camera.focus_distance if camera.focus_distance is not None else 0.5
    show_focus_overlay = True

    print(f"Displaying stream in window '{window_name}'. Press 'q' to quit.")

    # Display only supported controls
    if capabilities:
        available_controls = capabilities.get_available_controls()
        if available_controls:
            print("🎮 Available controls:")
            for control in available_controls:
                print(f"   {control}")
        else:
            print("⚠️  No manual camera controls available")
    else:
        print("🎯 Focus controls: 'a'=Auto focus, 'm'=Manual focus, '+/-'=Adjust manual focus")
        print(
            "💡 Exposure controls: 'e'=Toggle auto exposure, 'E'=Manual exposure, 'w/s'=Adjust exposure, 'r/t'=Adjust gain"
        )

    print("🔍 Debug: 'd'=Diagnostics, 'k'=Show key codes, 'f'=Toggle overlay")
    if record:
        print(f"🎬 Recording enabled - saving every {save_n_frames} frame(s) to: {output_dir}")

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is not None:
                # Check if this is a new frame by computing a simple hash
                current_frame_hash = hash(frame.tobytes())

                # Only process if we have a new frame
                if current_frame_hash != last_frame_hash:
                    # Add focus overlay to frame
                    display_frame = frame.copy()
                    if show_focus_overlay:
                        focus_mode = "AUTO" if not manual_focus_mode else f"MANUAL ({manual_focus_value:.2f})"
                        cv2.putText(
                            display_frame,
                            f"Focus: {focus_mode}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                        # Add exposure info if supported
                        if camera.cap and capabilities:
                            y_offset = 60
                            if capabilities.auto_exposure_supported or capabilities.manual_exposure_supported:
                                exposure = camera.cap.get(cv2.CAP_PROP_EXPOSURE)
                                auto_exposure = camera.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                                exposure_mode = "AUTO" if auto_exposure == 1 else f"MANUAL ({exposure:.2f})"
                                cv2.putText(
                                    display_frame,
                                    f"Exposure: {exposure_mode}",
                                    (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 255),
                                    2,
                                )
                                y_offset += 30

                            if capabilities.gain_supported:
                                gain = camera.cap.get(cv2.CAP_PROP_GAIN)
                                cv2.putText(
                                    display_frame,
                                    f"Gain: {gain:.2f}",
                                    (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 0, 255),
                                    2,
                                )
                                y_offset += 30

                            if capabilities.brightness_supported:
                                brightness = camera.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                                cv2.putText(
                                    display_frame,
                                    f"Brightness: {brightness:.2f}",
                                    (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 255, 0),
                                    2,
                                )
                                y_offset += 30

                        # Dynamic control text based on capabilities
                        if capabilities:
                            available_controls = capabilities.get_available_controls()
                            if available_controls:
                                control_text = " | ".join(available_controls)
                                cv2.putText(
                                    display_frame,
                                    control_text,
                                    (10, display_frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.35,
                                    (255, 255, 255),
                                    1,
                                )
                        else:
                            cv2.putText(
                                display_frame,
                                "Focus: a=Auto m=Manual +/-=Adjust | Exposure: e=Toggle E=Manual w/s=Adjust r/t=Gain",
                                (10, display_frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.35,
                                (255, 255, 255),
                                1,
                            )

                    cv2.imshow(window_name, display_frame)
                    frame_count += 1
                    last_frame_hash = current_frame_hash

                    # Save frame if recording is enabled
                    if record and frame_count % save_n_frames == 0 and output_dir is not None:
                        filename = output_dir / f"frame_{frame_count:06d}.jpg"
                        success = cv2.imwrite(str(filename), frame)
                        if success:
                            saved_count += 1

                    # Update status display every 0.5 seconds if recording
                    if record and start_time is not None and last_status_update is not None:
                        current_time = time.time()
                        if current_time - last_status_update >= 0.5:
                            elapsed = current_time - start_time
                            minutes = int(elapsed // 60)
                            seconds = elapsed % 60

                            if minutes > 0:
                                time_str = f"{minutes}m {seconds:.1f}s"
                            else:
                                time_str = f"{seconds:.1f}s"

                            print(
                                f"\r📸 Recording: {time_str} | Frames: {frame_count} | Saved: {saved_count}",
                                end="",
                                flush=True,
                            )
                            last_status_update = current_time
                else:
                    # Still need to update display even if frame hasn't changed
                    display_frame = frame.copy()
                    if show_focus_overlay:
                        focus_mode = "AUTO" if not manual_focus_mode else f"MANUAL ({manual_focus_value:.2f})"
                        cv2.putText(
                            display_frame,
                            f"Focus: {focus_mode}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                        # Add exposure info if supported
                        if camera.cap and capabilities:
                            y_offset = 60
                            if capabilities.auto_exposure_supported or capabilities.manual_exposure_supported:
                                exposure = camera.cap.get(cv2.CAP_PROP_EXPOSURE)
                                auto_exposure = camera.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                                exposure_mode = "AUTO" if auto_exposure == 1 else f"MANUAL ({exposure:.2f})"
                                cv2.putText(
                                    display_frame,
                                    f"Exposure: {exposure_mode}",
                                    (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 255),
                                    2,
                                )
                                y_offset += 30

                            if capabilities.gain_supported:
                                gain = camera.cap.get(cv2.CAP_PROP_GAIN)
                                cv2.putText(
                                    display_frame,
                                    f"Gain: {gain:.2f}",
                                    (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 0, 255),
                                    2,
                                )
                                y_offset += 30

                            if capabilities.brightness_supported:
                                brightness = camera.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                                cv2.putText(
                                    display_frame,
                                    f"Brightness: {brightness:.2f}",
                                    (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 255, 0),
                                    2,
                                )
                                y_offset += 30

                        # Dynamic control text based on capabilities
                        if capabilities:
                            available_controls = capabilities.get_available_controls()
                            if available_controls:
                                control_text = " | ".join(available_controls)
                                cv2.putText(
                                    display_frame,
                                    control_text,
                                    (10, display_frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.35,
                                    (255, 255, 255),
                                    1,
                                )
                        else:
                            cv2.putText(
                                display_frame,
                                "Focus: a=Auto m=Manual +/-=Adjust | Exposure: e=Toggle E=Manual w/s=Adjust r/t=Gain",
                                (10, display_frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.35,
                                (255, 255, 255),
                                1,
                            )
                    cv2.imshow(window_name, display_frame)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("a"):
                # Enable autofocus
                if capabilities and capabilities.autofocus_supported and camera.cap:
                    manual_focus_mode = False
                    camera.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                    print("✅ Autofocus enabled")
                else:
                    print("⚠️  Autofocus not supported by this camera")
            elif key == ord("m"):
                # Enable manual focus
                if capabilities and capabilities.manual_focus_supported and camera.cap:
                    manual_focus_mode = True
                    camera.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                    camera.cap.set(cv2.CAP_PROP_FOCUS, manual_focus_value)
                    print(f"🎯 Manual focus enabled at {manual_focus_value:.2f}")
                else:
                    print("⚠️  Manual focus not supported by this camera")
            elif key == ord("+") or key == ord("=") or key == 43 or key == 61:  # +/= keys (different keyboard layouts)
                # Increase manual focus (focus farther)
                if capabilities and capabilities.manual_focus_supported and manual_focus_mode and camera.cap:
                    old_value = manual_focus_value
                    min_focus, max_focus = capabilities.focus_range
                    step = (max_focus - min_focus) / 20  # 20 steps across the range
                    manual_focus_value = min(max_focus, manual_focus_value + step)
                    camera.cap.set(cv2.CAP_PROP_FOCUS, manual_focus_value)
                    actual_value = camera.cap.get(cv2.CAP_PROP_FOCUS)
                    print(f"🎯 Focus: {old_value:.2f} → {actual_value:.2f} (farther)")
                elif not capabilities or not capabilities.manual_focus_supported:
                    print("⚠️  Manual focus not supported by this camera")
                elif not manual_focus_mode:
                    print("⚠️  Switch to manual focus mode first (press 'm')")
            elif key == ord("-") or key == ord("_") or key == 45 or key == 95:  # -/_ keys (different keyboard layouts)
                # Decrease manual focus (focus nearer)
                if capabilities and capabilities.manual_focus_supported and manual_focus_mode and camera.cap:
                    old_value = manual_focus_value
                    min_focus, max_focus = capabilities.focus_range
                    step = (max_focus - min_focus) / 20  # 20 steps across the range
                    manual_focus_value = max(min_focus, manual_focus_value - step)
                    camera.cap.set(cv2.CAP_PROP_FOCUS, manual_focus_value)
                    actual_value = camera.cap.get(cv2.CAP_PROP_FOCUS)
                    print(f"🎯 Focus: {old_value:.2f} → {actual_value:.2f} (nearer)")
                elif not capabilities or not capabilities.manual_focus_supported:
                    print("⚠️  Manual focus not supported by this camera")
                elif not manual_focus_mode:
                    print("⚠️  Switch to manual focus mode first (press 'm')")
            elif key == ord("f"):
                # Toggle focus overlay
                show_focus_overlay = not show_focus_overlay
                print(f"🎯 Focus overlay {'enabled' if show_focus_overlay else 'disabled'}")
            elif key == ord("e"):
                # Toggle auto exposure
                if capabilities and capabilities.auto_exposure_supported and camera.cap:
                    current_auto_exposure = camera.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                    new_auto_exposure = 1 if current_auto_exposure == 0 else 0
                    camera.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, new_auto_exposure)
                    print(f"💡 Auto exposure {'enabled' if new_auto_exposure == 1 else 'disabled'}")
                else:
                    print("⚠️  Auto exposure toggle not supported by this camera")
            elif key == ord("E"):
                # Enable manual exposure mode
                if capabilities and capabilities.manual_exposure_supported and camera.cap:
                    camera.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
                    print("💡 Manual exposure mode enabled")
                else:
                    print("⚠️  Manual exposure mode not supported by this camera")
            elif key == ord("w"):
                # Decrease exposure (reduce light sensitivity)
                if capabilities and capabilities.manual_exposure_supported and camera.cap:
                    exposure = camera.cap.get(cv2.CAP_PROP_EXPOSURE)
                    min_exp, max_exp = capabilities.exposure_range
                    step = (max_exp - min_exp) / 20  # 20 steps across the range
                    new_exposure = max(min_exp, exposure - step)
                    camera.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                    actual_exposure = camera.cap.get(cv2.CAP_PROP_EXPOSURE)
                    print(f"💡 Exposure: {exposure:.2f} → {actual_exposure:.2f} (less light)")
                else:
                    print("⚠️  Manual exposure control not supported by this camera")
            elif key == ord("s"):
                # Increase exposure (increase light sensitivity)
                if capabilities and capabilities.manual_exposure_supported and camera.cap:
                    exposure = camera.cap.get(cv2.CAP_PROP_EXPOSURE)
                    min_exp, max_exp = capabilities.exposure_range
                    step = (max_exp - min_exp) / 20  # 20 steps across the range
                    new_exposure = min(max_exp, exposure + step)
                    camera.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                    actual_exposure = camera.cap.get(cv2.CAP_PROP_EXPOSURE)
                    print(f"💡 Exposure: {exposure:.2f} → {actual_exposure:.2f} (more light)")
                else:
                    print("⚠️  Manual exposure control not supported by this camera")
            elif key == ord("r"):
                # Decrease gain/ISO (reduce light sensitivity)
                if capabilities and capabilities.gain_supported and camera.cap:
                    gain = camera.cap.get(cv2.CAP_PROP_GAIN)
                    min_gain, max_gain = capabilities.gain_range
                    step = (max_gain - min_gain) / 20  # 20 steps across the range
                    new_gain = max(min_gain, gain - step)
                    camera.cap.set(cv2.CAP_PROP_GAIN, new_gain)
                    actual_gain = camera.cap.get(cv2.CAP_PROP_GAIN)
                    print(f"🎯 Gain: {gain:.2f} → {actual_gain:.2f} (less sensitive)")
                else:
                    print("⚠️  Gain control not supported by this camera")
            elif key == ord("t"):
                # Increase gain/ISO (increase light sensitivity)
                if capabilities and capabilities.gain_supported and camera.cap:
                    gain = camera.cap.get(cv2.CAP_PROP_GAIN)
                    min_gain, max_gain = capabilities.gain_range
                    step = (max_gain - min_gain) / 20  # 20 steps across the range
                    new_gain = min(max_gain, gain + step)
                    camera.cap.set(cv2.CAP_PROP_GAIN, new_gain)
                    actual_gain = camera.cap.get(cv2.CAP_PROP_GAIN)
                    print(f"🎯 Gain: {gain:.2f} → {actual_gain:.2f} (more sensitive)")
                else:
                    print("⚠️  Gain control not supported by this camera")
            elif key == ord("z"):
                # Decrease brightness
                if capabilities and capabilities.brightness_supported and camera.cap:
                    brightness = camera.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                    min_bright, max_bright = capabilities.brightness_range
                    step = (max_bright - min_bright) / 20  # 20 steps across the range
                    new_brightness = max(min_bright, brightness - step)
                    camera.cap.set(cv2.CAP_PROP_BRIGHTNESS, new_brightness)
                    actual_brightness = camera.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                    print(f"🔆 Brightness: {brightness:.2f} → {actual_brightness:.2f} (darker)")
                else:
                    print("⚠️  Brightness control not supported by this camera")
            elif key == ord("x"):
                # Increase brightness
                if capabilities and capabilities.brightness_supported and camera.cap:
                    brightness = camera.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                    min_bright, max_bright = capabilities.brightness_range
                    step = (max_bright - min_bright) / 20  # 20 steps across the range
                    new_brightness = min(max_bright, brightness + step)
                    camera.cap.set(cv2.CAP_PROP_BRIGHTNESS, new_brightness)
                    actual_brightness = camera.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                    print(f"🔆 Brightness: {brightness:.2f} → {actual_brightness:.2f} (brighter)")
                else:
                    print("⚠️  Brightness control not supported by this camera")
            elif key == ord("d"):
                # Diagnostic - test camera capabilities
                print("\n🔍 CAMERA DIAGNOSTICS")
                print("=" * 30)
                if camera.cap:
                    try:
                        # Test autofocus support
                        old_af = camera.cap.get(cv2.CAP_PROP_AUTOFOCUS)
                        camera.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                        new_af = camera.cap.get(cv2.CAP_PROP_AUTOFOCUS)
                        print(f"Autofocus support: {'✅ YES' if new_af == 1 else '❌ NO'} (set: 1, got: {new_af})")
                        camera.cap.set(cv2.CAP_PROP_AUTOFOCUS, old_af)  # restore

                        # Test manual focus support
                        old_focus = camera.cap.get(cv2.CAP_PROP_FOCUS)
                        print(f"Current focus value: {old_focus:.2f}")

                        # Try to set different focus values
                        camera.cap.set(cv2.CAP_PROP_FOCUS, 0.0)
                        near_focus = camera.cap.get(cv2.CAP_PROP_FOCUS)
                        camera.cap.set(cv2.CAP_PROP_FOCUS, 1.0)
                        far_focus = camera.cap.get(cv2.CAP_PROP_FOCUS)
                        camera.cap.set(cv2.CAP_PROP_FOCUS, old_focus)  # restore

                        print(f"Focus range test: 0.0 → {near_focus:.2f}, 1.0 → {far_focus:.2f}")
                        if abs(near_focus - far_focus) > 0.01:
                            print("Manual focus: ✅ YES (range detected)")
                        else:
                            print("Manual focus: ❌ NO (fixed focus camera)")

                        # Test exposure and other properties
                        exposure = camera.cap.get(cv2.CAP_PROP_EXPOSURE)
                        auto_exposure = camera.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                        gain = camera.cap.get(cv2.CAP_PROP_GAIN)
                        brightness = camera.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                        sharpness = camera.cap.get(cv2.CAP_PROP_SHARPNESS)

                        print(f"Exposure: {exposure:.2f} (Auto: {'ON' if auto_exposure == 1 else 'OFF'})")
                        print(f"Gain/ISO: {gain:.2f}")
                        print(f"Brightness: {brightness:.2f}")
                        print(f"Sharpness: {sharpness:.2f}")

                    except Exception as e:
                        print(f"Error during diagnostics: {e}")
                else:
                    print("❌ Camera not available for diagnostics")
                print("=" * 30)
            elif key == ord("k"):
                # Show key codes (for debugging keyboard issues)
                print(f"🔍 Key pressed: {key} (char: '{chr(key) if 32 <= key <= 126 else '?'}')")
                print("   Common key codes: +/=: 43/61, -/_: 45/95")
            elif key != 255:  # 255 is returned when no key is pressed
                # Show unknown key presses
                print(f"🔍 Unknown key: {key} (char: '{chr(key) if 32 <= key <= 126 else '?'}')")
                if capabilities:
                    available_controls = capabilities.get_available_controls()
                    controls_str = " | ".join(available_controls) if available_controls else "None available"
                    print(f"   Available controls: {controls_str}")
                    print("   Other keys: d (diagnostics), f (toggle overlay), q (quit)")
                else:
                    print(
                        "   Available keys: a/m/+/- (focus), e/E/w/s/r/t (exposure), z/x (brightness), d (diagnostics), f (toggle overlay), q (quit)"
                    )

    except KeyboardInterrupt:
        if record:
            print()  # New line after the dynamic status display
        print("\nInterrupted by user")
    finally:
        if record:
            print()  # New line after the dynamic status display
        cv2.destroyWindow(window_name)
        camera.stop_streaming()

        if record and start_time is not None:
            end_time = time.time()
            duration = end_time - start_time

            # Calculate timing statistics
            minutes = int(duration // 60)
            seconds = duration % 60

            if duration > 0:
                frames_per_second = saved_count / duration
                total_frames_captured = frame_count

                # Format duration string
                if minutes > 0:
                    duration_str = f"{minutes}m {seconds:.1f}s"
                else:
                    duration_str = f"{seconds:.1f}s"

                print(f"🎬 Recording complete: {saved_count} frames saved to {output_dir}")
                print(
                    f"⏱️  Duration: {duration_str} | Total frames: {total_frames_captured} | Saved rate: {frames_per_second:.2f} frames/sec"
                )
            else:
                print(f"🎬 Recording complete: {saved_count} frames saved to {output_dir}")


def main():
    """Main function."""
    args = parse_args()

    # Parse resolution
    try:
        width, height = map(int, args.resolution.split("x"))
    except ValueError:
        print(f"❌ Invalid resolution format: {args.resolution}")
        print("   Expected format: WIDTHxHEIGHT (e.g., 1280x720)")
        sys.exit(1)

    print("📹 Simple Camera Streaming")
    print("=" * 30)
    print(f"🎥 Camera: {args.camera}")
    print(f"📐 Resolution: {width}x{height}")
    print(f"🎯 FPS: {args.fps}")
    print(f"🪟 Window: '{args.window_name}'")

    if args.record:
        output_dir = create_output_directory(args.output_dir)
        print("🎬 Recording: Enabled")
        print(f"📁 Output Directory: {output_dir}")
        print(f"🎞️  Save Rate: Every {args.save_n_frames} frame(s)")
    else:
        output_dir = None
        print("🎬 Recording: Disabled")

    print()
    print("📝 Controls:")
    print("   - Press 'q' in the video window to quit")
    print("   - Or press Ctrl+C in terminal")
    print()

    # Determine focus settings
    enable_autofocus = not args.disable_autofocus and args.manual_focus is None
    focus_distance = args.manual_focus

    # Create camera instance with enhanced focus settings
    camera = WebcamCapture(
        camera_index=args.camera,
        width=width,
        height=height,
        fps=args.fps,
        enable_autofocus=enable_autofocus,
        focus_distance=focus_distance,
    )

    # Connect to camera
    print(f"🔌 Connecting to camera {args.camera}...")
    if not camera.connect():
        print(f"❌ Failed to connect to camera {args.camera}")
        print("💡 Try:")
        print("   - Check if camera is connected")
        print("   - Try a different camera index (--camera 1, --camera 2, etc.)")
        print("   - Make sure no other app is using the camera")
        sys.exit(1)

    print("✅ Camera connected successfully!")

    # Get and display actual camera FPS
    width, height, actual_fps = camera.get_frame_info()
    print(f"📊 Actual camera settings: {width}x{height} @ {actual_fps:.1f} FPS")

    # Display initial focus settings
    focus_mode = (
        "Autofocus" if enable_autofocus else f"Manual ({focus_distance:.2f})" if focus_distance else "Manual (0.5)"
    )
    print(f"🎯 Focus mode: {focus_mode}")

    # Test camera capabilities
    print()
    capabilities = CameraCapabilities(camera.cap)
    print()

    # If test-capabilities flag is set, show detailed info and exit
    if args.test_capabilities:
        print("🔬 DETAILED CAMERA CAPABILITIES")
        print("=" * 40)
        print(f"Autofocus: {'✅ Supported' if capabilities.autofocus_supported else '❌ Not supported'}")
        print(f"Manual focus: {'✅ Supported' if capabilities.manual_focus_supported else '❌ Not supported'}")
        if capabilities.manual_focus_supported:
            print(f"   Focus range: {capabilities.focus_range[0]:.2f} - {capabilities.focus_range[1]:.2f}")
        print(f"Auto exposure: {'✅ Supported' if capabilities.auto_exposure_supported else '❌ Not supported'}")
        print(f"Manual exposure: {'✅ Supported' if capabilities.manual_exposure_supported else '❌ Not supported'}")
        if capabilities.manual_exposure_supported:
            print(f"   Exposure range: {capabilities.exposure_range[0]:.2f} - {capabilities.exposure_range[1]:.2f}")
        print(f"Gain control: {'✅ Supported' if capabilities.gain_supported else '❌ Not supported'}")
        if capabilities.gain_supported:
            print(f"   Gain range: {capabilities.gain_range[0]:.2f} - {capabilities.gain_range[1]:.2f}")
        print(f"Brightness control: {'✅ Supported' if capabilities.brightness_supported else '❌ Not supported'}")
        if capabilities.brightness_supported:
            print(
                f"   Brightness range: {capabilities.brightness_range[0]:.2f} - {capabilities.brightness_range[1]:.2f}"
            )
        print(f"Sharpness control: {'✅ Supported' if capabilities.sharpness_supported else '❌ Not supported'}")
        print("=" * 40)
        camera.disconnect()
        return

    try:
        # Start streaming with optional recording
        print("🚀 Starting video stream...")
        display_stream_with_recording(
            camera, args.window_name, args.record, output_dir, args.save_n_frames, capabilities
        )

    except KeyboardInterrupt:
        print("\n⏹️  Stream interrupted by user")

    except Exception as e:
        print(f"❌ Error during streaming: {e}")
        sys.exit(1)

    finally:
        print("🧹 Cleaning up...")
        camera.disconnect()
        print("✅ Done!")


if __name__ == "__main__":
    main()
