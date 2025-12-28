import numpy as np

from aeropiper.models.robots.manipulators.manipulator_model import ManipulatorModel
from aeropiper.utils.mjcf_utils import xml_path_completion


class AeroPiper(ManipulatorModel):
    """
    AeroPiper is a fixed-frame bimanual robot with two 6-DoF arms and two tendon-driven hands.
    """

    arms = ["right", "left"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/aeropiper/robot.xml"), idn=idn)

    @property
    def default_base(self):
        # This robot already includes its own fixed frame. Don't add an external mount.
        return "NullMount"

    @property
    def default_gripper(self):
        return {"right": "AeroPiperRightHand", "left": "AeroPiperLeftHand"}

    @property
    def default_controller_config(self):
        # Not used for composite controller loading (we provide controllers/config/robots/default_aeropiper.json),
        # but required by the interface.
        return {"right": "default_aeropiper", "left": "default_aeropiper"}

    @property
    def init_qpos(self):
        # 6 right arm joints + 6 left arm joints
        # Start in a neutral-ish configuration (can be tuned later)
        return np.array(
            [
                0.0,
                1.57,
                -1.35,
                0.0,
                0.0,
                0.0,
                0.0,
                1.57,
                -1.35,
                0.0,
                0.0,
                0.0,
            ]
        )

    @property
    def base_xpos_offset(self):
        # Rough offsets to align AeroPiper above typical robosuite arenas
        # These may be tuned once we validate reachability in each task.
        #
        # User-requested global shift vs previous placement:
        # - 15cm further back (negative x)
        # - 15cm higher (positive z)
        dx = -0.15
        dz = 0.15
        return {
            "bins": (-0.35 + dx, 0.0, 0.0 + dz),
            "empty": (-0.35 + dx, 0.0, 0.0 + dz),
            "table": lambda table_length: (-0.20 + dx - table_length / 2, 0.0, 0.0 + dz),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.8

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        # These bodies exist in robots/aeropiper/robot.xml and are where grippers are merged
        return {"right": "right_hand", "left": "left_hand"}


