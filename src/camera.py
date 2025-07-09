#!/usr/bin/env python3
"""
Simple webcam capture class for loading frames and streaming video.
"""

import threading
import time

import cv2
import numpy as np


class WebcamCapture:
    """
    A simple webcam capture class that can connect to a webcam and capture frames.

    Features:
    - Connect to webcam by index (usually 0 for default camera)
    - Load single frames
    - Stream video with background thread
    - Configure resolution and FPS
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_autofocus: bool = True,
        focus_distance: float | None = None,
    ):
        """
        Initialize the webcam capture.

        Args:
            camera_index: Camera device index (0 for default camera)
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
            enable_autofocus: Enable autofocus (True) or manual focus (False)
            focus_distance: Manual focus distance (0.0 to 1.0, only used if enable_autofocus=False)
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_autofocus = enable_autofocus
        self.focus_distance = focus_distance

        self.cap: cv2.VideoCapture | None = None
        self.is_streaming = False
        self.stream_thread: threading.Thread | None = None
        self.latest_frame: np.ndarray | None = None
        self.frame_lock = threading.Lock()

    def connect(self) -> bool:
        """
        Connect to the webcam.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False

            # Set basic camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Configure autofocus and image quality settings
            self._configure_image_quality()

            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            print(f"Camera connected: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
            return True

        except Exception as e:
            print(f"Error connecting to camera: {e}")
            return False

    def _configure_image_quality(self):
        """Configure camera settings for optimal image quality and reduced blur."""
        if self.cap is None:
            return

        # Configure autofocus
        if self.enable_autofocus:
            # Enable autofocus
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            print("Autofocus enabled")
        else:
            # Disable autofocus and set manual focus
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            if self.focus_distance is not None:
                # Set manual focus (0.0 = near, 1.0 = far)
                self.cap.set(cv2.CAP_PROP_FOCUS, self.focus_distance)
                print(f"Manual focus set to {self.focus_distance}")

        # Try to optimize other settings for sharpness
        # Note: Not all cameras support all these properties
        try:
            # Enable auto-exposure for better lighting adaptation
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure

            # Set sharpness if supported (higher values = sharper)
            self.cap.set(cv2.CAP_PROP_SHARPNESS, 1.0)

            # Disable auto white balance for more consistent colors
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)

            # Set exposure compensation to reduce motion blur
            # Lower values = shorter exposure = less motion blur
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)

        except Exception:
            # Some cameras don't support all properties - this is expected
            pass

    def is_connected(self) -> bool:
        """
        Check if camera is connected.

        Returns:
            bool: True if connected, False otherwise
        """
        return self.cap is not None and self.cap.isOpened()

    def load_frame(self) -> np.ndarray | None:
        """
        Capture and return a single frame from the webcam.

        Returns:
            np.ndarray: Frame as BGR image array, or None if failed
        """
        if not self.is_connected():
            print("Error: Camera not connected")
            return None

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            return None

        return frame

    def start_streaming(self) -> bool:
        """
        Start streaming video in a background thread.

        Returns:
            bool: True if streaming started successfully, False otherwise
        """
        if not self.is_connected():
            print("Error: Camera not connected")
            return False

        if self.is_streaming:
            print("Already streaming")
            return True

        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        print("Video streaming started")
        return True

    def stop_streaming(self):
        """Stop video streaming."""
        if self.is_streaming:
            self.is_streaming = False
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=2.0)
            print("Video streaming stopped")

    def get_latest_frame(self) -> np.ndarray | None:
        """
        Get the latest frame from the streaming thread.

        Returns:
            np.ndarray: Latest frame as BGR image array, or None if no frame available
        """
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def _stream_loop(self):
        """Internal streaming loop that runs in background thread."""
        while self.is_streaming and self.is_connected():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                print("Warning: Failed to capture frame in streaming loop")
                time.sleep(0.01)  # Small delay to prevent busy loop

    def display_stream(self, window_name: str = "Webcam Stream") -> None:
        """
        Display the video stream in a window. Press 'q' to quit.

        Args:
            window_name: Name of the display window
        """
        if not self.start_streaming():
            return

        print(f"Displaying stream in window '{window_name}'. Press 'q' to quit.")

        try:
            while True:
                frame = self.get_latest_frame()
                if frame is not None:
                    cv2.imshow(window_name, frame)

                # Check for 'q' key press to quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cv2.destroyWindow(window_name)
            self.stop_streaming()

    def save_frame(self, filename: str) -> bool:
        """
        Save current frame to file.

        Args:
            filename: Path to save the image file

        Returns:
            bool: True if saved successfully, False otherwise
        """
        frame = self.load_frame()
        if frame is not None:
            success = cv2.imwrite(filename, frame)
            if success:
                print(f"Frame saved to {filename}")
            return success
        return False

    def get_frame_info(self) -> tuple[int, int, float]:
        """
        Get current frame dimensions and FPS.

        Returns:
            Tuple[int, int, float]: (width, height, fps)
        """
        if not self.is_connected():
            return (0, 0, 0.0)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        return (width, height, fps)

    def disconnect(self):
        """Disconnect from the webcam and clean up resources."""
        self.stop_streaming()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()
        print("Camera disconnected")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def test_focus_settings(self):
        """
        Interactive method to test different focus settings and find optimal configuration.
        Press keys to adjust focus in real-time.
        """
        if not self.is_connected():
            print("Error: Camera not connected")
            return

        print("\n🎯 Focus Testing Mode")
        print("===================")
        print("Controls:")
        print("  'a' - Enable autofocus")
        print("  'm' - Switch to manual focus")
        print("  '+' - Increase manual focus (focus farther)")
        print("  '-' - Decrease manual focus (focus nearer)")
        print("  'r' - Reset to default settings")
        print("  'q' - Quit focus testing")
        print()

        if not self.start_streaming():
            return

        manual_focus = 0.5  # Start with middle focus distance

        try:
            while True:
                frame = self.get_latest_frame()
                if frame is not None:
                    # Add focus info overlay
                    overlay = frame.copy()
                    mode = "AUTO" if self.enable_autofocus else f"MANUAL ({manual_focus:.2f})"
                    cv2.putText(overlay, f"Focus: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(
                        overlay,
                        "Press 'a'=Auto, 'm'=Manual, '+/-'=Adjust, 'q'=Quit",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                    cv2.imshow("Focus Test", overlay)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("a"):
                    # Enable autofocus
                    self.enable_autofocus = True
                    if self.cap:
                        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                    print("✅ Autofocus enabled")
                elif key == ord("m"):
                    # Switch to manual focus
                    self.enable_autofocus = False
                    if self.cap:
                        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                        self.cap.set(cv2.CAP_PROP_FOCUS, manual_focus)
                    print(f"🎯 Manual focus set to {manual_focus:.2f}")
                elif key == ord("+") or key == ord("="):
                    # Increase focus distance (focus farther)
                    if not self.enable_autofocus:
                        manual_focus = min(1.0, manual_focus + 0.1)
                        if self.cap:
                            self.cap.set(cv2.CAP_PROP_FOCUS, manual_focus)
                        print(f"🎯 Focus increased to {manual_focus:.2f}")
                elif key == ord("-"):
                    # Decrease focus distance (focus nearer)
                    if not self.enable_autofocus:
                        manual_focus = max(0.0, manual_focus - 0.1)
                        if self.cap:
                            self.cap.set(cv2.CAP_PROP_FOCUS, manual_focus)
                        print(f"🎯 Focus decreased to {manual_focus:.2f}")
                elif key == ord("r"):
                    # Reset to default
                    self.enable_autofocus = True
                    manual_focus = 0.5
                    self._configure_image_quality()
                    print("🔄 Reset to default settings")

        except KeyboardInterrupt:
            print("\nFocus testing interrupted")
        finally:
            cv2.destroyWindow("Focus Test")
            self.stop_streaming()

            # Apply the final settings
            self.focus_distance = manual_focus if not self.enable_autofocus else None
            print(
                f"\n✅ Final settings: {'Autofocus' if self.enable_autofocus else f'Manual focus at {manual_focus:.2f}'}"
            )


def main():
    """Example usage of the WebcamCapture class."""
    print("WebcamCapture Example")
    print("====================")

    # Example 1: Basic frame capture
    print("\n1. Basic frame capture:")
    camera = WebcamCapture(camera_index=0, width=640, height=480, fps=30)

    if camera.connect():
        # Load and save a single frame
        frame = camera.load_frame()
        if frame is not None:
            print(f"Captured frame shape: {frame.shape}")
            camera.save_frame("captured_frame.jpg")

        # Get camera info
        width, height, fps = camera.get_frame_info()
        print(f"Camera info: {width}x{height} @ {fps:.1f} FPS")

        camera.disconnect()

    # Example 2: Using context manager for streaming
    print("\n2. Video streaming with context manager:")
    print("Press 'q' in the video window to quit")

    with WebcamCapture(camera_index=0) as cam:
        if cam.is_connected():
            cam.display_stream("My Webcam")


if __name__ == "__main__":
    main()
