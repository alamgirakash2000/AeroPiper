"""
Camera Detection Tool

Helps identify which camera is at which index.
Opens each available camera and shows the feed.

Usage:
    python stereo_control/detect_cameras.py
"""

import cv2
import sys


def detect_cameras(max_cameras=5):
    """Detect and display available cameras."""
    print("\n" + "="*60)
    print("CAMERA DETECTION")
    print("="*60)
    print("Checking camera indices 0-4...")
    print("Press 'Q' or ESC in any window to continue\n")
    
    available_cameras = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Windows DirectShow
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Get camera info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps
                })
                
                print(f"[OK] Camera {i} found: {width}x{height} @ {fps:.1f} FPS")
                
                # Show preview
                window_name = f"Camera {i} - Press Q/ESC to continue"
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Add label
                    cv2.putText(
                        frame,
                        f"CAMERA INDEX: {i}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA
                    )
                    cv2.putText(
                        frame,
                        f"{width}x{height} @ {fps:.1f}fps",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )
                    cv2.putText(
                        frame,
                        "Press Q or ESC",
                        (20, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA
                    )
                    
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord('q'), 27]:  # Q or ESC
                        break
                
                cv2.destroyWindow(window_name)
            cap.release()
        else:
            print(f"[--] Camera {i} not available")
    
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if len(available_cameras) < 2:
        print(f"WARNING: Only {len(available_cameras)} camera(s) detected")
        print("You need at least 2 cameras for stereo tracking")
        return None
    
    print(f"[OK] Found {len(available_cameras)} cameras:")
    for cam in available_cameras:
        print(f"  Index {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']:.1f} FPS")
    
    print("\nFor stereo tracking, you'll use two of these.")
    print("Recommended: Skip the laptop's built-in webcam")
    print("\nExample commands:")
    
    # Find the two external cameras (usually higher indices or better resolution)
    if len(available_cameras) >= 3:
        # Likely indices 1 and 2 (0 is laptop cam)
        print(f"  # If camera 0 is laptop webcam, use 1 and 2:")
        print(f"  python stereo_control/create_default_calibration.py --left 1 --right 2")
    else:
        print(f"  python stereo_control/create_default_calibration.py --left 0 --right 1")
    
    return available_cameras


if __name__ == "__main__":
    try:
        cameras = detect_cameras()
        if cameras and len(cameras) >= 2:
            print("\n[OK] You're ready for stereo tracking!")
            print("\nNext step: Create default calibration")
            print("  python stereo_control/create_default_calibration.py")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        cv2.destroyAllWindows()
        sys.exit(1)

