#!/usr/bin/env python3
"""
Simple Reach Task
Just run: python3 task_reach.py
"""

import time
from robot_connection import (
    initialize_robots, 
    move_arm, 
    set_hand, 
    go_home, 
    shutdown_robots,
    OPEN, 
    PINCH,
    piper,
    hand
)

# Task positions
APPROACH = [15, 25, -20, 0, 18, 0]
GRASP_POS = [15, 40, -35, 0, 28, 0]
LIFT = [15, 20, -15, 0, 15, 0]

def main():
    print("\n" + "=" * 70)
    print("TASK: REACH AND GRASP")
    print("=" * 70)
    
    # Wake up robots
    import robot_connection
    robot_connection.piper, robot_connection.hand = initialize_robots()
    
    if not robot_connection.piper or not robot_connection.hand:
        return False
    
    # Do task
    print("\n[1] Approach...")
    set_hand(OPEN, "Open")
    move_arm(APPROACH)
    
    print("\n[2] Grasp...")
    move_arm(GRASP_POS)
    set_hand(PINCH, "Pinch")
    
    print("\n[3] Lift...")
    move_arm(LIFT)
    
    print("\n[4] Home...")
    shutdown_robots()
    
    return True

if __name__ == "__main__":
    main()