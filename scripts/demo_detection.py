#!/usr/bin/env python3
"""
Demo script for Clash Royale game state detection.

Usage:
    # Capture from Bluestacks and detect
    python scripts/demo_detection.py

    # Detect from an image file
    python scripts/demo_detection.py --image path/to/screenshot.png

    # Save visualization
    python scripts/demo_detection.py --save output.png

    # Continuous capture mode
    python scripts/demo_detection.py --continuous
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# Colors for visualization (BGR format)
FRIENDLY_COLOR = (0, 255, 0)    # Green
ENEMY_COLOR = (0, 0, 255)       # Red
NEUTRAL_COLOR = (255, 255, 0)   # Cyan


def draw_detections(detector):
    """Draw bounding boxes with labels on the arena image."""
    # Use arena image (part2) since that's where detections are made
    arena = detector.get_last_arena()
    if arena is None:
        return None

    # Make a copy to draw on
    vis = arena.copy()

    # Get detection data
    detections = detector.get_detections()

    boxes = detections.get('boxes', [])
    confidences = detections.get('confidences', [])
    class_names = detections.get('class_names', [])
    belongs = detections.get('belongs', [])

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        conf = confidences[i] if i < len(confidences) else 0
        name = class_names[i] if i < len(class_names) else 'unknown'
        bel = belongs[i] if i < len(belongs) else -1

        # Choose color based on belonging
        if bel == 0:
            color = FRIENDLY_COLOR
        elif bel == 1:
            color = ENEMY_COLOR
        else:
            color = NEUTRAL_COLOR

        # Draw box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Draw label with confidence
        label = f"{name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Background for label
        cv2.rectangle(vis, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)

        # Text
        cv2.putText(vis, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return vis


def main():
    parser = argparse.ArgumentParser(description='Clash Royale Game State Detection Demo')
    parser.add_argument('--image', '-i', type=str, help='Path to screenshot image')
    parser.add_argument('--save', '-s', type=str, help='Save visualization to file')
    parser.add_argument('--continuous', '-c', action='store_true',
                       help='Continuous capture mode')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Capture interval in seconds (for continuous mode)')
    parser.add_argument('--no-tracking', action='store_true',
                       help='Disable object tracking')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')
    parser.add_argument('--no-ocr', action='store_true',
                       help='Disable OCR (faster startup, no time detection)')
    parser.add_argument('--no-cards', action='store_true',
                       help='Disable card detection')
    parser.add_argument('--show', action='store_true',
                       help='Show OpenCV window with detection visualization')
    args = parser.parse_args()

    print("Initializing Clash Royale Game State Detector...")
    print("This may take a moment to load models...")

    try:
        from cr_detection.state import GameStateDetector
    except ImportError as e:
        print(f"\nError importing detection module: {e}")
        print("\nMake sure to install dependencies first:")
        print("  cd /Users/will/Documents/Projects/CReinforcement-Learning")
        print("  source venv/bin/activate")
        print("  pip install -r requirements.txt")
        return 1

    # Create detector
    detector = GameStateDetector(
        use_tracking=not args.no_tracking,
        use_gpu=not args.no_gpu,
        use_ocr=not args.no_ocr,
        use_cards=not args.no_cards
    )
    print("Detector initialized!")

    if args.image:
        # Detect from image file
        print(f"\nDetecting from image: {args.image}")
        try:
            state = detector.detect_from_file(args.image)
            print_state(state)

            if args.save:
                detector.save_visualization(args.save)
                print(f"\nVisualization saved to: {args.save}")

            if args.show:
                vis = draw_detections(detector)
                if vis is not None:
                    cv2.imshow('CR Detection', vis)
                    print("\nPress any key to close window...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error processing image: {e}")
            return 1

    elif args.continuous:
        # Continuous capture mode
        print("\nStarting continuous capture mode...", flush=True)
        print("Press Ctrl+C to stop (or 'q' in window if --show)", flush=True)
        print("-" * 50, flush=True)

        try:
            while True:
                state = detector.capture_and_detect()
                if state is None:
                    print("Failed to capture screenshot. Is Bluestacks running?", flush=True)
                else:
                    print_state_compact(state)

                    if args.show:
                        vis = draw_detections(detector)
                        if vis is not None:
                            cv2.imshow('CR Detection', vis)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                break

                time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            if args.show:
                cv2.destroyAllWindows()

    else:
        # Single capture
        print("\nCapturing screenshot from Bluestacks...")
        state = detector.capture_and_detect()

        if state is None:
            print("Failed to capture screenshot.")
            print("\nMake sure Bluestacks is running and visible on screen.")
            return 1

        print_state(state)

        if args.save:
            detector.save_visualization(args.save)
            print(f"\nVisualization saved to: {args.save}")

    return 0


def print_state(state):
    """Print detailed game state information."""
    print("\n" + "=" * 50)
    print("GAME STATE")
    print("=" * 50)

    print(f"\nTime: {state.time:.1f}s")
    print(f"Frame: {state.frame_count}")

    friendly = state.get_friendly_units()
    enemy = state.get_enemy_units()

    print(f"\nFriendly Units ({len(friendly)}):")
    if friendly:
        for unit in friendly:
            pos = f"({unit.xy[0]:.1f}, {unit.xy[1]:.1f})" if unit.xy is not None else "(?)"
            print(f"  - {unit.class_name} at {pos}")
    else:
        print("  (none)")

    print(f"\nEnemy Units ({len(enemy)}):")
    if enemy:
        for unit in enemy:
            pos = f"({unit.xy[0]:.1f}, {unit.xy[1]:.1f})" if unit.xy is not None else "(?)"
            print(f"  - {unit.class_name} at {pos}")
    else:
        print("  (none)")

    if state.towers:
        print("\nTowers:")
        for key, tower in state.towers.items():
            side = "Friendly" if tower['bel'] == 0 else "Enemy"
            print(f"  - {tower['type']} ({side})")

    if state.cards:
        print("\nHand Cards:")
        for card in state.cards:
            print(f"  Slot {card.slot}: {card.name} ({card.confidence:.2f})")

    print("\n" + "=" * 50)


def print_state_compact(state, verbose=True):
    """Print compact state summary."""
    friendly = state.get_friendly_units()
    enemy = state.get_enemy_units()
    timestamp = time.strftime("%H:%M:%S")

    # Cards summary
    cards_str = ""
    if state.cards:
        card_names = [c.name for c in state.cards if c.name != 'unknown']
        if card_names:
            cards_str = f" | Cards: {', '.join(card_names)}"

    # Main summary line
    print(f"[{timestamp}] Time: {state.time:5.1f}s | Friendly: {len(friendly):2d} | Enemy: {len(enemy):2d}{cards_str}", flush=True)

    if verbose and (friendly or enemy):
        # Show detected units
        if friendly:
            names = [u.class_name for u in friendly]
            print(f"  Friendly: {', '.join(names)}", flush=True)
        if enemy:
            names = [u.class_name for u in enemy]
            print(f"  Enemy: {', '.join(names)}", flush=True)


if __name__ == '__main__':
    sys.exit(main())
