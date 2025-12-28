from .robot_model import RobotModel, create_robot
from .manipulators import *


def is_robosuite_robot(robot: str) -> bool:
    """
    robot is robosuite repo robot if can import robot class from aeropiper.models.robots
    """
    try:
        module = __import__("aeropiper.models.robots", fromlist=[robot])
        getattr(module, robot)
        return True
    except (ImportError, AttributeError):
        return False
