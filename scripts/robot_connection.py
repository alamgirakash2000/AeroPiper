#!/usr/bin/env python3
"""
Robot Connection and Testing
=============================
Just plug in and run: python3 robot_connection.py

The script will automatically request sudo if needed for CAN setup.

Hand gestures have been refined based on tested trajectory values for better
accuracy and natural movement.
"""

import time
import sys
import os
import subprocess
from piper_sdk import C_PiperInterface_V2
from aero_open_sdk.aero_hand import AeroHand

# ============================================
# CONFIGURATION
# ============================================

HAND_PORT = "/dev/ttyACM0"
HAND_BAUDRATE = 921600

def deg_to_units(degrees):
    return round(degrees * 1000)

# Positions (degrees)
HOME = [0, 0, 0, 0, 0, 20]
POINT_A = [10, 20, -15, 0, 15, 35]
POINT_B = [-15, 30, -25, 0, 20, 0]
POINT_C = [20, 35, -30, 5, 25, -10]
POINT_D = [-10, 25, -20, -5, 18, 30]

# Gestures (7 DOF) - Tuned values from working trajectory
# Format: [j0, j1, j2, j3_index, j4_middle, j5_ring, j6_pinkie]
OPEN = [0, 0, 0, 0, 0, 0, 0]
PINCH_INDEX = [75, 25, 30, 50, 0, 0, 0]  # Index finger pinch
PINCH_MIDDLE = [83, 42, 23, 0, 50, 0, 0]  # Middle finger pinch
PINCH_RING = [100, 42, 23, 0, 0, 52, 0]  # Ring finger pinch
PINCH_PINKIE = [100, 35, 23, 0, 0, 0, 50]  # Pinkie finger pinch
PEACE = [90, 45, 60, 0, 0, 90, 90]  # Index & middle extended, rest closed
ROCKSTAR = [0, 0, 0, 0, 90, 90, 0]  # Index & pinkie extended, middle & ring closed
POINT = [60, 30, 60, 0, 60, 60, 60]  # Only index extended
THUMBS = [0, 0, 0, 90, 90, 90, 90]  # Thumb up, all fingers closed
FIST = [90, 50, 90, 90, 90, 90, 90]  # All closed tight

piper = None
hand = None

# ============================================
# AUTOMATIC CAN SETUP
# ============================================

def setup_can_with_sudo():
    """Setup CAN automatically - will ask for sudo password if needed"""
    print("\n" + "=" * 70)
    print("SETTING UP CAN INTERFACE")
    print("=" * 70)
    
    try:
        # Check if already UP
        result = subprocess.run(["ip", "link", "show", "can0"], 
                              capture_output=True, text=True)
        if "UP" in result.stdout:
            print("✓ CAN already configured and ready!")
            return True
        
        print("\n→ CAN needs configuration...")
        print("→ Requesting sudo access (you may need to enter password)...\n")
        
        # Run CAN setup commands with sudo
        # The user will be prompted for password if needed
        commands = [
            ["sudo", "ip", "link", "set", "can0", "down"],
            ["sudo", "ip", "link", "set", "can0", "type", "can", "bitrate", "1000000"],
            ["sudo", "ip", "link", "set", "can0", "up"]
        ]
        
        for cmd in commands:
            subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        
        # Verify
        result = subprocess.run(["ip", "link", "show", "can0"], 
                              capture_output=True, text=True)
        
        if "UP" in result.stdout:
            print("✓ CAN interface configured successfully!")
            return True
        else:
            print("✗ CAN setup failed")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ CAN setup failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def setup_serial_permissions():
    """Setup serial port permissions automatically"""
    print("\n" + "=" * 70)
    print("CHECKING SERIAL PORT ACCESS")
    print("=" * 70)
    
    try:
        # Check if port exists
        if not os.path.exists(HAND_PORT):
            print(f"✗ Serial port {HAND_PORT} not found!")
            print(f"→ Please check if the hand is connected")
            print(f"→ Try: ls /dev/ttyACM* /dev/ttyUSB*")
            return False
        
        # Try to open the port to check permissions
        try:
            # Test if we can access it
            with open(HAND_PORT, 'r') as f:
                pass
            print(f"✓ Serial port {HAND_PORT} is accessible!")
            return True
        except PermissionError:
            # Need to fix permissions
            print(f"\n→ Serial port needs permission fix...")
            print("→ Requesting sudo access (you may need to enter password)...\n")
            
            # Fix permissions with sudo
            subprocess.run(["sudo", "chmod", "666", HAND_PORT], check=True)
            
            # Verify
            try:
                with open(HAND_PORT, 'r') as f:
                    pass
                print("✓ Serial port permissions fixed!")
                return True
            except:
                print("✗ Permission fix failed")
                return False
                
    except subprocess.CalledProcessError as e:
        print(f"✗ Permission setup failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

# ============================================
# INITIALIZE
# ============================================

def init_piper():
    global piper
    print("\n→ PiPER Arm...")
    piper = C_PiperInterface_V2()
    piper.ConnectPort()
    print("    ✓ Connected")
    
    while not piper.EnablePiper():
        time.sleep(0.01)
    time.sleep(0.5)
    print("    ✓ Enabled")
    return piper

def init_hand():
    global hand
    print("\n→ Aero Hand...")
    hand = AeroHand(port=HAND_PORT, baudrate=HAND_BAUDRATE)
    print(f"    ✓ Connected")
    print("    → Initializing...")
    time.sleep(1.0)  # Give hand time to be ready
    print("    ✓ Ready")
    return hand

def initialize_robots():
    """Initialize both robots - can be imported by other scripts"""
    print("\n" + "=" * 70)
    print("INITIALIZING ROBOTS")
    print("=" * 70)
    
    # Setup CAN automatically
    if not setup_can_with_sudo():
        return None, None
    
    # Setup serial port permissions automatically
    if not setup_serial_permissions():
        return None, None
    
    p = init_piper()
    if not p:
        return None, None
    
    h = init_hand()
    if not h:
        return None, None
    
    print("\n✓ ALL READY\n")
    return p, h

# ============================================
# MOVEMENT FUNCTIONS (for import)
# ============================================

def move_arm(pos, speed=15, wait=3.5, desc=""):
    """Move arm - can be imported"""
    if desc:
        print(f"\n→ {desc}")
    
    j = [deg_to_units(d) for d in pos]
    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    time.sleep(0.05)
    piper.JointCtrl(j[0], j[1], j[2], j[3], j[4], j[5])
    time.sleep(wait)
    print("    ✓ Done")

def set_hand(gest, name=""):
    """Set hand gesture - can be imported"""
    if name:
        print(f"→ {name}")
    g16 = hand.convert_seven_joints_to_sixteen(gest)
    hand.set_joint_positions(g16)
    time.sleep(2)
    print("    ✓ Done")

def go_home():
    """Go home - can be imported"""
    print("\n" + "=" * 70)
    print("GOING HOME")
    print("=" * 70)
    set_hand(OPEN, "Open hand")
    move_arm(HOME, 12, 4, "Home position")
    print("✓ Ready for next task\n")

def demo_finger_sequence():
    """Demo individual finger pinches - can be imported"""
    print("\n" + "=" * 70)
    print("FINGER PINCH DEMO")
    print("=" * 70)
    
    set_hand(OPEN, "Open hand")
    time.sleep(0.5)
    
    fingers = [
        (PINCH_INDEX, "Touch Index"),
        (PINCH_MIDDLE, "Touch Middle"),
        (PINCH_RING, "Touch Ring"),
        (PINCH_PINKIE, "Touch Pinkie"),
    ]
    
    for gest, name in fingers:
        set_hand(gest, name)
        time.sleep(0.5)
        set_hand(OPEN, "Release")
        time.sleep(0.3)
    
    print("✓ Finger demo complete\n")

def shutdown_robots():
    """Cleanup - can be imported"""
    print("\n→ Disconnecting...")
    go_home()
    hand.close()
    print("✓ DONE\n")

# ============================================
# TEST SEQUENCE
# ============================================

def run_tests():
    """Main test sequence"""
    print("=" * 70)
    print("TEST SEQUENCE")
    print("=" * 70)
    print("\n⚠️  Robot will move!")
    
    for i in [3, 2, 1]:
        print(f"Starting in {i}...")
        time.sleep(1)
    
    tests = [
        (POINT_A, PINCH_INDEX, "Point A", "Index Pinch"),
        (POINT_B, PEACE, "Point B", "Peace Sign"),
        (POINT_C, ROCKSTAR, "Point C", "Rock On"),
        (POINT_D, POINT, "Point D", "Pointing"),
        (POINT_A, THUMBS, "Point A", "Thumbs Up"),
        (POINT_B, FIST, "Point B", "Fist"),
    ]
    
    for i, (pos, gest, pname, gname) in enumerate(tests, 1):
        print(f"\n{'=' * 70}")
        print(f"TEST {i}/{len(tests)}: {pname} + {gname}")
        print(f"{'=' * 70}")
        
        set_hand(OPEN, "Open (safety)")
        move_arm(pos, desc=pname)
        set_hand(gest, gname)
        time.sleep(1)
    
    print("\n✓ ALL TESTS COMPLETE")

# ============================================
# MAIN
# ============================================

def main():
    print("\n" + "=" * 70)
    print("ROBOT CONNECTION & TEST")
    print("=" * 70)
    
    try:
        # Initialize (will handle CAN automatically)
        global piper, hand
        piper, hand = initialize_robots()
        
        if not piper or not hand:
            return False
        
        # Run tests
        run_tests()
        
        # Cleanup
        shutdown_robots()
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  STOPPED!")
        try:
            if hand:
                set_hand(OPEN, "Emergency")
                hand.close()
        except:
            pass
        return False
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)