"""
Bluestacks screenshot capture for macOS.

Uses Quartz for efficient window capture of the Bluestacks emulator.
"""

import numpy as np
from typing import Optional, Tuple
import subprocess


def get_bluestacks_window_id() -> Optional[int]:
    """
    Find the Bluestacks window ID on macOS.

    Returns:
        Window ID as integer, or None if not found.
    """
    try:
        # Use AppleScript to find Bluestacks window
        script = '''
        tell application "System Events"
            set blueStacksProcess to first process whose name contains "Bluestacks"
            set blueStacksWindow to first window of blueStacksProcess
            return id of blueStacksWindow
        end tell
        '''
        result = subprocess.run(['osascript', '-e', script],
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except Exception:
        pass
    return None


def capture_bluestacks_screenshot() -> Optional[np.ndarray]:
    """
    Capture a screenshot of the Bluestacks window.

    Returns:
        Screenshot as numpy array in BGR format, or None if capture fails.
    """
    try:
        import Quartz
        import Quartz.CoreGraphics as CG

        # Get list of windows
        window_list = CG.CGWindowListCopyWindowInfo(
            CG.kCGWindowListOptionOnScreenOnly,
            CG.kCGNullWindowID
        )

        # Find Bluestacks window
        bluestacks_window = None
        for window in window_list:
            owner_name = window.get('kCGWindowOwnerName', '')
            window_name = window.get('kCGWindowName', '')
            if 'bluestacks' in owner_name.lower() or 'bluestacks' in window_name.lower():
                bluestacks_window = window
                break

        if bluestacks_window is None:
            print("Bluestacks window not found")
            return None

        # Get window bounds
        bounds = bluestacks_window.get('kCGWindowBounds', {})
        x = int(bounds.get('X', 0))
        y = int(bounds.get('Y', 0))
        width = int(bounds.get('Width', 0))
        height = int(bounds.get('Height', 0))

        if width == 0 or height == 0:
            print("Invalid window dimensions")
            return None

        # Capture the specific window by ID
        image = CG.CGWindowListCreateImage(
            CG.CGRectNull,  # Null rect = capture entire window
            CG.kCGWindowListOptionIncludingWindow,  # Capture this specific window
            bluestacks_window['kCGWindowNumber'],
            CG.kCGWindowImageBoundsIgnoreFraming
        )

        if image is None:
            print("Failed to capture window image")
            return None

        # Convert to numpy array
        width = CG.CGImageGetWidth(image)
        height = CG.CGImageGetHeight(image)
        bytes_per_row = CG.CGImageGetBytesPerRow(image)

        data_provider = CG.CGImageGetDataProvider(image)
        data = CG.CGDataProviderCopyData(data_provider)

        # Create numpy array from raw data
        arr = np.frombuffer(data, dtype=np.uint8)
        arr = arr.reshape((height, bytes_per_row // 4, 4))
        arr = arr[:, :width, :]  # Remove padding

        # Convert BGRA to BGR
        bgr = arr[:, :, :3]

        return bgr

    except ImportError:
        print("Quartz not available. Install pyobjc-framework-Quartz.")
        return None
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None


def capture_with_pyautogui() -> Optional[np.ndarray]:
    """
    Fallback capture using pyautogui (captures entire screen).

    Returns:
        Screenshot as numpy array in BGR format.
    """
    try:
        import pyautogui
        screenshot = pyautogui.screenshot()
        img = np.array(screenshot)
        # Convert RGB to BGR
        return img[:, :, ::-1]
    except Exception as e:
        print(f"Error with pyautogui capture: {e}")
        return None


class BluestacksCapture:
    """
    Screenshot capture class for Bluestacks emulator.

    Supports both Quartz (preferred) and pyautogui (fallback) methods.
    """

    def __init__(self, use_quartz: bool = True):
        """
        Initialize capture.

        Args:
            use_quartz: Use Quartz for capture (faster, window-specific).
        """
        self.use_quartz = use_quartz
        self._last_screenshot: Optional[np.ndarray] = None

    def capture(self) -> Optional[np.ndarray]:
        """
        Capture a screenshot from Bluestacks.

        Returns:
            Screenshot as numpy array in BGR format.
        """
        if self.use_quartz:
            screenshot = capture_bluestacks_screenshot()
            if screenshot is not None:
                self._last_screenshot = screenshot
                return screenshot

        # Fallback to pyautogui
        screenshot = capture_with_pyautogui()
        if screenshot is not None:
            self._last_screenshot = screenshot
        return screenshot

    def get_last_screenshot(self) -> Optional[np.ndarray]:
        """Get the last captured screenshot."""
        return self._last_screenshot

    def get_window_size(self) -> Optional[Tuple[int, int]]:
        """
        Get the Bluestacks window size.

        Returns:
            Tuple of (width, height) or None if not found.
        """
        try:
            import Quartz.CoreGraphics as CG

            window_list = CG.CGWindowListCopyWindowInfo(
                CG.kCGWindowListOptionOnScreenOnly,
                CG.kCGNullWindowID
            )

            for window in window_list:
                owner_name = window.get('kCGWindowOwnerName', '')
                if 'bluestacks' in owner_name.lower():
                    bounds = window.get('kCGWindowBounds', {})
                    return (int(bounds.get('Width', 0)),
                           int(bounds.get('Height', 0)))
        except Exception:
            pass
        return None


def save_screenshot(image: np.ndarray, path: str) -> bool:
    """
    Save a screenshot to file.

    Args:
        image: Screenshot as numpy array (BGR format).
        path: Output file path.

    Returns:
        True if successful.
    """
    try:
        import cv2
        cv2.imwrite(path, image)
        return True
    except Exception as e:
        print(f"Error saving screenshot: {e}")
        return False
