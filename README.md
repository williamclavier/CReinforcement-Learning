# Clash Royale Reinforcement Learning

Game state detection for Clash Royale using computer vision, designed for reinforcement learning applications.

## Features

- **Unit Detection**: Detects 150+ unit types on the battlefield using YOLOv8
- **Tower Health**: OCR-based tower health detection
- **Game Time**: Timer detection via OCR
- **Object Tracking**: ByteTrack for consistent unit tracking across frames
- **Bluestacks Integration**: Automated screenshot capture on macOS

## Installation

1. Create and activate virtual environment:
```bash
cd /Users/will/Documents/Projects/CReinforcement-Learning
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from cr_detection.state import GameStateDetector

# Initialize detector
detector = GameStateDetector()

# Capture from Bluestacks and detect
state = detector.capture_and_detect()

print(f"Game time: {state.time}s")
print(f"Friendly units: {len(state.get_friendly_units())}")
print(f"Enemy units: {len(state.get_enemy_units())}")

# Or detect from an image file
state = detector.detect_from_file("screenshot.png")
```

## Demo Script

```bash
# Capture from Bluestacks
python scripts/demo_detection.py

# Detect from image file
python scripts/demo_detection.py --image path/to/screenshot.png

# Continuous capture mode
python scripts/demo_detection.py --continuous

# Save visualization
python scripts/demo_detection.py --save output.png
```

## Project Structure

```
CReinforcement-Learning/
├── src/cr_detection/
│   ├── capture/          # Bluestacks screenshot capture
│   ├── models/           # YOLOv8 detector and tracking
│   ├── ocr/              # PaddleOCR for text recognition
│   ├── processing/       # Image splitting and constants
│   ├── state/            # State building and high-level API
│   └── utils/            # Utility functions
├── models/               # Model weight files
├── scripts/              # Demo and utility scripts
└── tests/                # Test files
```

## Resolution Support

Currently supported aspect ratios:
- 2.22 (1080x2400) - Primary
- 2.16 (592x1280)
- 2.13 (600x1280)
- 1.78 (1080x1920) - Bluestacks 16:9 (placeholder params, may need calibration)

### Calibrating for Your Resolution

If detection isn't working correctly, you may need to calibrate the split parameters for your Bluestacks resolution:

1. Take a screenshot during gameplay
2. Use `calibrate_bbox_params()` to visualize regions
3. Adjust parameters in `src/cr_detection/processing/constants.py`

## Credits

Detection models and core algorithms ported from [KataCR](https://github.com/wty-yy/KataCR).
