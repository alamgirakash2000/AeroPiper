# AeroPiper: A Dual-Hand Manipulation System

<p align="center">
  <img src="images/demo.gif" alt="Demo" width="90%"/>
</p>

## Images

<p align="center">
  <img src="images/piper.png" alt="PiPER Arm" width="45%"/>
  <img src="images/aero_hand_open.png" alt="TetherIA Aero Hand" width="45%"/>
  <br/>
</p>

### Description
AeroPiper is a dual-hand manipulation system that combines two AgileX PiPER 6‑DOF robotic arms with two TetherIA Aero Open hands, targeting dexterous, human-like manipulation tasks. The project pairs high-fidelity MuJoCo simulation assets with reinforcement-learning baselines so you can prototype, train, and evaluate complex bimanual skills rapidly—then transfer them to real hardware.

### Official resources
- **TetherIA Aero Hand Open Docs**: `https://docs.tetheria.ai`
- **AgileX PiPER product page**: `https://global.agilex.ai/products/piper`
- **MuJoCo documentation**: `https://mujoco.readthedocs.io`

---

## Installation

### Using pip (recommended)
```bash
# Clone the repository
git clone https://github.com/alamgirakash2000/AeroPiper
cd AeroPiper

# Install dependencies
pip install -r requirements.txt

# For MuJoCo viewer support (optional but recommended)
pip install 'mujoco[glfw]'
```

### Using conda
```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate AeroPiper

# For MuJoCo viewer support (optional but recommended)
pip install 'mujoco[glfw]'
```

### Verify Installation
```bash
python -c "import torch; import mujoco; import numpy; print('Installation successful!')"
```

---

<p style="color:red"><strong>This system is currently under development. For now, only training of the Pick & Place task is available.</strong></p>

## Quick Start

### Train (Pick & Place)
```bash
python scripts/train.py --task pick_place
# Add --randomize for harder random cube/target positions
```

### Evaluate
```bash
python scripts/eval_pick_place.py --checkpoint checkpoints/pick_place/YOUR_RUN/model_final.pt
```

---

## Action Space (6D)

```
action = [arm_select, j1, j2, j3, j4, j5]
```

| Index | Description |
|-------|-------------|
| 0 | `arm_select`: >= 0 for right arm, < 0 for left arm |
| 1-5 | Joint controls for joints 1-5 |
| (6) | Joint 6 is always held at 0 |

---

## Training Options

### Default Values (no flags needed)
| Parameter | Default |
|-----------|---------|
| `--num-envs` | 64 |
| `--iterations` | 10000 |
| `--lr` | 1e-4 |
| `--max-episode-steps` | 500 |
| `--device` | cuda |
| Curriculum | Disabled (train reach+place together) |

### Optional Flags

| Flag | Description |
|------|-------------|
| `--randomize` | Random cube/target positions each episode |
| `--curriculum` | Enable curriculum (learn reach first, then place) |
| `--resume PATH` | Resume from checkpoint |
| `--iterations N` | Override iteration count |
| `--lr X` | Override learning rate |
| `--num-envs N` | Override number of parallel environments |
| `--max-episode-steps N` | Override episode length |


## Evaluation Options

### Default Values
| Parameter | Default |
|-----------|---------|
| `--episodes` | 5 |
| `--max-steps` | 500 |
| Movement | Realtime (smooth) |
| Actions | Stochastic |

### Optional Flags

| Flag | Description |
|------|-------------|
| `--randomize` | Random positions each episode |
| `--deterministic` | Use deterministic actions |
| `--fast` | Fast mode (skip smooth movement) |
| `--no-viewer` | Run without visualization |
| `--episodes N` | Number of episodes |
| `--max-steps N` | Max steps per episode |

### Examples

**Basic evaluation:**
```bash
python scripts/eval_pick_place.py --checkpoint YOUR_MODEL.pt
```

**Test generalization with random positions:**
```bash
python scripts/eval_pick_place.py --checkpoint YOUR_MODEL.pt --randomize --episodes 10
```

**Fast headless evaluation:**
```bash
python scripts/eval_pick_place.py --checkpoint YOUR_MODEL.pt --fast --no-viewer --episodes 100
```

---
