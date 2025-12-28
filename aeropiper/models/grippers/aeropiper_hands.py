"""
AeroPiper dexterous hands (right / left).

These are based on the user's AeroPiper tendon-driven hands and are exposed as robosuite GripperModels so that
existing manipulation environments can use grasp checks and controllers without environment changes.
"""

import numpy as np

from aeropiper.models.grippers.gripper_model import GripperModel
from aeropiper.utils.mjcf_utils import xml_path_completion


class AeroPiperRightHand(GripperModel):
    """
    AeroPiper right hand.

    Notes:
    - This gripper has 7 actuators (1 thumb abduction joint + 6 tendon position actuators).
    - The hand contains many finger joints; we initialize all joint qpos to zeros by default.
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/aeropiper_right_hand.xml"), idn=idn)
        self._init_qpos = np.zeros(len(self.joints))

    def format_action(self, action):
        # Expose all 7 controls directly (no remapping)
        assert len(action) == self.dof
        return np.array(action)

    @property
    def init_qpos(self):
        return self._init_qpos

    @property
    def _important_geoms(self):
        # Use thumb tip as left_fingerpad and all other fingertips as right_fingerpad.
        # These names correspond to collision geoms in the hand MJCF.
        return {
            "left_finger": ["th_tip"],
            "right_finger": ["if_tip", "mf_tip", "rf_tip", "pf_tip"],
            "left_fingerpad": ["th_tip"],
            "right_fingerpad": ["if_tip", "mf_tip", "rf_tip", "pf_tip"],
        }


class AeroPiperLeftHand(GripperModel):
    """
    AeroPiper left hand.
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/aeropiper_left_hand.xml"), idn=idn)
        self._init_qpos = np.zeros(len(self.joints))

    def format_action(self, action):
        assert len(action) == self.dof
        return np.array(action)

    @property
    def init_qpos(self):
        return self._init_qpos

    @property
    def _important_geoms(self):
        # Left hand has left_* names for fingertip collision geoms
        return {
            "left_finger": ["left_th_tip"],
            "right_finger": ["left_if_tip", "left_mf_tip", "left_rf_tip", "left_pf_tip"],
            "left_fingerpad": ["left_th_tip"],
            "right_fingerpad": ["left_if_tip", "left_mf_tip", "left_rf_tip", "left_pf_tip"],
        }


