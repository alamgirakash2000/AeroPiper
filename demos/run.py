import os
import sys

import numpy as np
import aeropiper as suite
from aeropiper.utils.action_print_utils import print_robot_action_bounds

# Allow running as `python demos/run.py` (ensure repo root is on sys.path)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

env = suite.make(
    env_name="PickPlace",
    robots="AeroPiper",
    env_configuration="single-robot",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    renderer="mjviewer",
    render_camera=None,
    ignore_done=True,
)

env.reset()
env.viewer.set_camera(camera_id=-1)

# Prints ONLY: "<action_index> <action_name> <min> <max>"
#print_robot_action_bounds(env)

low, high = env.action_spec
for _ in range(4000):
    action = np.random.uniform(low, high)
    obs, reward, done, info = env.step(action)
    env.render()