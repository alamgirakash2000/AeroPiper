"""
Local `aeropiper` package for this simplified repository.

We import the environment modules so their classes get registered into
`suite.ALL_ENVIRONMENTS` (registration happens at import-time via metaclasses).
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Import env modules so all selectable tasks are registered.
# -----------------------------------------------------------------------------

from aeropiper.environments.base import make  # noqa: E402

# Import manipulation env modules (these define 18 registered env classes total)
from aeropiper.environments.manipulation.lift import Lift  # noqa: F401,E402
from aeropiper.environments.manipulation.stack import Stack  # noqa: F401,E402
from aeropiper.environments.manipulation.nut_assembly import (  # noqa: F401,E402
    NutAssembly,
    NutAssemblySingle,
    NutAssemblySquare,
    NutAssemblyRound,
)
from aeropiper.environments.manipulation.pick_place import (  # noqa: F401,E402
    PickPlace,
    PickPlaceSingle,
    PickPlaceMilk,
    PickPlaceBread,
    PickPlaceCereal,
    PickPlaceCan,
)
from aeropiper.environments.manipulation.door import Door  # noqa: F401,E402
from aeropiper.environments.manipulation.wipe import Wipe  # noqa: F401,E402
from aeropiper.environments.manipulation.tool_hang import ToolHang  # noqa: F401,E402
from aeropiper.environments.manipulation.two_arm_lift import TwoArmLift  # noqa: F401,E402
from aeropiper.environments.manipulation.two_arm_peg_in_hole import TwoArmPegInHole  # noqa: F401,E402
from aeropiper.environments.manipulation.two_arm_handover import TwoArmHandover  # noqa: F401,E402
from aeropiper.environments.manipulation.two_arm_transport import TwoArmTransport  # noqa: F401,E402

from aeropiper.environments import ALL_ENVIRONMENTS  # noqa: E402

# Common registries / helpers used by the entrypoints
from aeropiper.controllers import (  # noqa: E402
    ALL_PART_CONTROLLERS,
    load_part_controller_config,
    ALL_COMPOSITE_CONTROLLERS,
    load_composite_controller_config,
)
from aeropiper.robots import ALL_ROBOTS  # noqa: E402
from aeropiper.models.grippers import ALL_GRIPPERS  # noqa: E402
from aeropiper.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER  # noqa: E402

__version__ = "1.5.1"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\\  [~]\\/_   |//  |
     ] [   OOO      /o|__|
"""


