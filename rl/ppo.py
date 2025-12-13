"""
Proximal Policy Optimization (PPO) Algorithm
=============================================
"""

import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mujoco

# Optional native Mujoco viewer (requires mujoco with GLFW support)
try:
    import mujoco.viewer as mj_viewer  # type: ignore
    HAVE_MJ_VIEWER = True
except Exception:  # pragma: no cover - optional dependency
    mj_viewer = None
    HAVE_MJ_VIEWER = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from rl.networks import ActorCritic


@dataclass
class PPOConfig:
    """PPO training configuration."""
    # Training
    num_iterations: int = 5000
    num_steps_per_env: int = 64
    num_envs: int = 128
    
    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    
    # Learning rate
    learning_rate: float = 3e-4
    lr_schedule: str = "adaptive"  # "fixed", "linear", "adaptive"
    
    # Mini-batch
    num_mini_batches: int = 4
    num_epochs: int = 5
    
    # Network
    hidden_dims: list = None
    activation: str = "elu"
    init_noise_std: float = 1.0
    # Success replay / behavior cloning assist
    success_buffer_capacity: int = 5000
    bc_coef: float = 0.02
    bc_sample_size: int = 512
    bc_store_reach: bool = False
    # Curriculum: keep place phase off until reach success hits target
    curriculum_place_start: int = -1  # disabled when < 0
    curriculum_reach_threshold: float = 0.90
    curriculum_reach_window: int = 500  # iterations window for reach avg
    
    # Logging
    log_interval: int = 10
    save_interval: int = 500
    
    # Device
    device: str = "cuda"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 256]


class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        
        # Allocate buffers
        self.observations = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)
        
        self.step = 0

    def add(self, obs, action, reward, done, value, log_prob):
        """Add a transition to the buffer."""
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.log_probs[self.step] = log_prob
        self.step += 1

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        """Compute returns and GAE advantages."""
        last_gae = 0
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[step + 1]
            
            next_non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            self.advantages[step] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        self.returns = self.advantages + self.values

    def get_minibatch_generator(self, num_mini_batches: int):
        """Generate mini-batches for training."""
        batch_size = self.num_envs * self.num_steps
        mini_batch_size = batch_size // num_mini_batches
        
        # Flatten data
        obs = self.observations.view(-1, self.observations.shape[-1])
        actions = self.actions.view(-1, self.actions.shape[-1])
        values = self.values.view(-1)
        log_probs = self.log_probs.view(-1)
        advantages = self.advantages.view(-1)
        returns = self.returns.view(-1)
        
        # Random permutation
        indices = torch.randperm(batch_size, device=self.device)
        
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            batch_indices = indices[start:end]
            
            yield {
                "obs": obs[batch_indices],
                "actions": actions[batch_indices],
                "values": values[batch_indices],
                "log_probs": log_probs[batch_indices],
                "advantages": advantages[batch_indices],
                "returns": returns[batch_indices],
            }
    
    def reset(self):
        """Reset buffer for new rollout."""
        self.step = 0


class SuccessReplay:
    """Simple success replay buffer for auxiliary behavior cloning."""

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, obs: torch.Tensor, action: torch.Tensor):
        # Detach and store on device
        self.buffer.append((obs.detach().to(self.device), action.detach().to(self.device)))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if len(self.buffer) == 0:
            return None
        n = min(batch_size, len(self.buffer))
        idx = torch.randint(0, len(self.buffer), (n,), device=self.device)
        obs_batch = torch.stack([self.buffer[i][0] for i in idx])
        act_batch = torch.stack([self.buffer[i][1] for i in idx])
        return obs_batch, act_batch


class PPO:
    """
    PPO algorithm with wandb logging.
    
    Args:
        env: Vectorized environment
        config: PPO configuration
        reward_fn: Optional custom reward function (for reward shaping)
    """
    
    def __init__(
        self,
        env,
        config: PPOConfig,
        task_name: str = "unknown",
        reward_fn: Optional[callable] = None,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        render_window: bool = False,
        render_interval: int = 1,
    ):
        self.env = env
        self.config = config
        self.task_name = task_name
        self.reward_fn = reward_fn
        self.device = torch.device(config.device)
        self.render_window = render_window
        self.render_interval = max(1, render_interval)
        self.viewer = None
        self.use_wandb = False  # default; updated below if wandb enabled
        # Training state
        self.iteration = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.reach_successes = []
        self.place_successes = []
        self.reach_distances = []
        self.place_distances = []
        self.success_replay = SuccessReplay(
            capacity=config.success_buffer_capacity, device=self.device
        )
        
        # Get dimensions from env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.num_envs = config.num_envs
        
        # Create actor-critic network
        self.policy = ActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            init_noise_std=config.init_noise_std,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            num_envs=config.num_envs,
            num_steps=config.num_steps_per_env,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
        )

        # Optional Mujoco viewer for live visualization (env 0)
        self._vis_base_env = None
        self._last_distance_print = 0
        if self.render_window:
            if HAVE_MJ_VIEWER:
                try:
                    base_env = getattr(self.env.envs[0], "_env", self.env.envs[0])
                    self._vis_base_env = base_env
                    
                    # Add custom visualization sites to the model
                    self._add_distance_visualization_sites(base_env.model, base_env.data)
                    
                    self.viewer = mj_viewer.launch_passive(base_env.model, base_env.data)
                    print("Mujoco viewer launched (env 0) with palm-object distance visualization enabled.")
                    print("  Green sphere = Palm position | Blue sphere = Object position | Red line = Distance")
                except Exception as exc:  # pragma: no cover - visualization utility
                    self.viewer = None
                    print(f"Warning: failed to launch Mujoco viewer: {exc}")
            else:
                self.viewer = None
                print("Warning: mujoco.viewer not available; install mujoco with GLFW support (pip install 'mujoco[glfw]' or apt-get libs).")

        # Initialize wandb
        self.use_wandb = wandb_project is not None and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name or f"{task_name}_{time.strftime('%Y%m%d_%H%M%S')}",
                config={
                    "task": task_name,
                    "obs_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    **vars(config),
                },
            )
            wandb.watch(self.policy, log="all", log_freq=100)

        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self._total_episodes = 0

    def _add_geom(self, scn, geom_type, pos=None, size=None, rgba=None, fromto=None):
        """Append a debug geom to the scene (best-effort; avoids overflow)."""
        if scn.ngeom >= scn.maxgeom:
            return
        g = mujoco.MjvGeom()
        mujoco.mjv_initGeom(
            g,
            geom_type,
            np.zeros(3),
            np.zeros(3),
            np.ones(3),
            None,
        )
        if pos is not None:
            g.pos[:] = pos
        if size is not None:
            g.size[: len(size)] = size
        if rgba is not None:
            g.rgba[:] = rgba
        if fromto is not None:
            g.fromto[:] = fromto
        scn.geoms[scn.ngeom] = g
        scn.ngeom += 1
    
    def _add_distance_visualization_sites(self, model, data):
        """Initialize visualization by storing reference to model and data."""
        # The passive viewer will call its own rendering, we'll inject visuals each frame
        pass
    
    def _draw_distance_line(self):
        """Draw line and markers between palm and object using MuJoCo's connector API."""
        if self.viewer is None or self._vis_base_env is None:
            return
        
        try:
            base_env = self._vis_base_env
            
            # Check if it's a pick_place task
            if not hasattr(base_env, "cube_site_id"):
                return
            
            # Get positions
            cube_pos = base_env.data.site_xpos[base_env.cube_site_id].copy()
            
            # Get active arm end-effector
            if hasattr(self.env.envs[0], "_active_arm"):
                active_arm = self.env.envs[0]._active_arm
            else:
                active_arm = "right"
            
            ee_site_name = base_env.ee_sites.get(active_arm, "grasp_site")
            ee_site_id = base_env._name_to_id(mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
            palm_pos = base_env.data.site_xpos[ee_site_id].copy()
            
            # Calculate distance
            distance = np.linalg.norm(palm_pos - cube_pos)
            
            # Use mjv_connector to draw line - this is the proper MuJoCo API
            # Access the viewer's internal scene
            if hasattr(self.viewer, 'user_scn'):
                scn = self.viewer.user_scn
                # Add connector (line) between two points
                mujoco.mjv_connector(
                    scn,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    3,  # width in pixels
                    palm_pos[0], palm_pos[1], palm_pos[2],
                    cube_pos[0], cube_pos[1], cube_pos[2]
                )
                
                # Add spheres at endpoints using mjv_initGeom
                # Palm sphere (green)
                self._add_geom(
                    scn,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    pos=palm_pos,
                    size=np.array([0.02, 0, 0]),
                    rgba=np.array([0.0, 1.0, 0.0, 0.9])
                )
                
                # Object sphere (blue)
                self._add_geom(
                    scn,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    pos=cube_pos,
                    size=np.array([0.02, 0, 0]),
                    rgba=np.array([0.0, 0.0, 1.0, 0.9])
                )
            
            # Console output
            self._last_distance_print += 1
            if self._last_distance_print >= 20:
                print(f"\n  [DISTANCE VISUALIZATION]")
                print(f"  Palm position (GREEN sphere): [{palm_pos[0]:.3f}, {palm_pos[1]:.3f}, {palm_pos[2]:.3f}]")
                print(f"  Object position (BLUE sphere): [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
                print(f"  Distance (RED line): {distance:.4f} m")
                print(f"  Active arm: {active_arm.upper()}\n")
                self._last_distance_print = 0
                
        except Exception as e:
            if not hasattr(self, '_vis_error_printed'):
                import traceback
                print(f"\n[Visualization Warning] {e}")
                print(traceback.format_exc())
                self._vis_error_printed = True
    
    def collect_rollout(self, obs: torch.Tensor) -> torch.Tensor:
        """Collect rollout data from environment."""
        self.buffer.reset()
        
        for step in range(self.config.num_steps_per_env):
            with torch.no_grad():
                action, log_prob, value = self.policy.act(obs)
            
            # Clip action to valid range
            action_clipped = torch.clamp(action, -1.0, 1.0)
            
            # Step environment (VecEnv returns obs, reward, terminated, truncated, infos)
            step_out = self.env.step(action_clipped)
            if len(step_out) == 4:
                # Backward compatibility: obs, reward, done, info
                next_obs, reward, done, info = step_out
                terminated = done
                truncated = torch.zeros_like(done, dtype=torch.bool)
            else:
                next_obs, reward, terminated, truncated, info = step_out
                done = terminated | truncated

            # Live viewer update (env 0 shares data with viewer)
            if self.viewer is not None and (step % self.render_interval == 0):
                try:
                    # Add visualization BEFORE sync
                    self._draw_distance_line()
                    self.viewer.sync()
                except Exception as exc:  # pragma: no cover - visualization utility
                    print(f"Viewer sync failed, disabling viewer: {exc}")
                    self.viewer = None
            
            # Apply custom reward shaping if provided
            if self.reward_fn is not None:
                reward = self.reward_fn(obs, action, next_obs, reward, done, info)
            
            # Store transition
            self.buffer.add(obs, action, reward, done.float(), value, log_prob)
            
            # Track episode stats and intermediate progress
            self._track_episode_stats(reward, done, info)
            self._track_intermediate_progress(info)
            self._maybe_store_success(obs, action, info)
            
            obs = next_obs
            self.total_steps += self.num_envs
        
        # Compute returns and advantages
        with torch.no_grad():
            last_value = self.policy.get_value(obs)
        self.buffer.compute_returns_and_advantages(
            last_value, self.config.gamma, self.config.gae_lambda
        )
        
        return obs
    
    def _track_episode_stats(self, reward, done, info):
        """Track episode statistics."""
        # Track step-level rewards for immediate feedback
        if not hasattr(self, '_step_rewards'):
            self._step_rewards = []
        if isinstance(reward, torch.Tensor):
            self._step_rewards.extend(reward.cpu().numpy().tolist())
        else:
            self._step_rewards.append(float(reward))
        
        # Initialize episode-level tracking lists
        if not hasattr(self, '_episode_reach_successes'):
            self._episode_reach_successes = []
        if not hasattr(self, '_episode_place_successes'):
            self._episode_place_successes = []
        if not hasattr(self, '_episode_sequential_successes'):
            self._episode_sequential_successes = []
        
        # For vectorized env, info might be a list
        if isinstance(info, list):
            for i, inf in enumerate(info):
                if done[i]:
                    self._total_episodes += 1  # Count completed episodes
                    if "episode_reward" in inf:
                        self.episode_rewards.append(inf["episode_reward"])
                    if "episode_length" in inf:
                        self.episode_lengths.append(inf["episode_length"])
                    if "success" in inf:
                        self.episode_successes.append(float(inf["success"]))
                    
                    # Track EPISODE-LEVEL success (was reach/place achieved at any point?)
                    ep_reach = inf.get("success_reach", False)
                    ep_place = inf.get("success_place", False)
                    ep_sequential = inf.get("success", False)  # True only if both achieved
                    
                    self._episode_reach_successes.append(1 if ep_reach else 0)
                    self._episode_place_successes.append(1 if ep_place else 0)
                    self._episode_sequential_successes.append(1 if ep_sequential else 0)

    def _track_intermediate_progress(self, info):
        """Track reach/place success and distances for progress visibility."""
        if not isinstance(info, list):
            return
        for inf in info:
            if not isinstance(inf, dict):
                continue
            if "success_reach" in inf:
                self.reach_successes.append(float(inf["success_reach"]))
            if "success_place" in inf:
                self.place_successes.append(float(inf["success_place"]))
            if "dist_cube_hand" in inf:
                self.reach_distances.append(float(inf["dist_cube_hand"]))
            # For place distance, use palm-to-target if available (curriculum wrapper)
            if "dist_palm_target" in inf:
                self.place_distances.append(float(inf["dist_palm_target"]))
            elif "dist_cube_target" in inf:
                self.place_distances.append(float(inf["dist_cube_target"]))
            
            # Track curriculum phase for pick_place
            if "curriculum_phase" in inf:
                self._current_phase = inf["curriculum_phase"]
            # Track episode-level phase (more important for sequential learning)
            if "episode_phase" in inf:
                if not hasattr(self, '_episode_phases'):
                    self._episode_phases = []
                self._episode_phases.append(inf["episode_phase"])
            if "reach_improved" in inf:
                if not hasattr(self, '_improvements'):
                    self._improvements = []
                self._improvements.append(float(inf["reach_improved"]))
            if "reach_reverted" in inf:
                if not hasattr(self, '_reverts'):
                    self._reverts = []
                self._reverts.append(float(inf["reach_reverted"]))
            if "place_improved" in inf:
                if not hasattr(self, '_place_improvements'):
                    self._place_improvements = []
                self._place_improvements.append(float(inf["place_improved"]))
            # Track when both reach AND place are achieved in same step (sequential success)
            # Use a list to track recent successes for better rate calculation
            if not hasattr(self, '_sequential_success_list'):
                self._sequential_success_list = []
            # Record 1 if both, 0 otherwise
            sr = inf.get("success_reach", False)
            sp = inf.get("success_place", False)
            both_success = sr and sp
            self._sequential_success_list.append(1 if both_success else 0)

    def _maybe_store_success(self, obs: torch.Tensor, action: torch.Tensor, info):
        """Store successful transitions for auxiliary BC training."""
        if self.config.bc_coef <= 0:
            return
        if not isinstance(info, list):
            return
        for idx, inf in enumerate(info):
            if not isinstance(inf, dict):
                continue
            reached = inf.get("success_reach", False)
            placed = inf.get("success", False) or inf.get("success_place", False)
            # Only store placements by default; optionally store reach if enabled.
            if placed or (self.config.bc_store_reach and reached):
                self.success_replay.add(obs[idx], action[idx])
    
    def update(self) -> Dict[str, float]:
        """Perform PPO update."""
        # Normalize advantages
        advantages = self.buffer.advantages.view(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages.view(self.config.num_steps_per_env, self.num_envs)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        total_bc_loss = 0
        num_updates = 0
        
        for epoch in range(self.config.num_epochs):
            for batch in self.buffer.get_minibatch_generator(self.config.num_mini_batches):
                # Evaluate actions
                log_prob, value, entropy = self.policy.evaluate(batch["obs"], batch["actions"])
                
                # Policy loss (PPO clip)
                ratio = torch.exp(log_prob - batch["log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * batch["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = ((value - batch["returns"]) ** 2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Optional BC loss from success replay
                bc_loss = torch.tensor(0.0, device=self.device)
                if self.config.bc_coef > 0:
                    bc_batch = self.success_replay.sample(self.config.bc_sample_size)
                    if bc_batch is not None:
                        bc_obs, bc_actions = bc_batch
                        bc_log_prob, _, _ = self.policy.evaluate(bc_obs, bc_actions)
                        bc_loss = -bc_log_prob.mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                    + self.config.bc_coef * bc_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                with torch.no_grad():
                    kl = (batch["log_probs"] - log_prob).mean()
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    total_kl += kl.item()
                    total_bc_loss += bc_loss.item()
                    num_updates += 1
        
        # Learning rate scheduling
        if self.config.lr_schedule == "adaptive":
            avg_kl = total_kl / num_updates
            if avg_kl > 0.02:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * 0.95, 1e-5)
            elif avg_kl < 0.005:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = min(param_group['lr'] * 1.05, 1e-2)
        
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "kl": total_kl / num_updates,
            "bc_loss": total_bc_loss / num_updates,
            "lr": self.optimizer.param_groups[0]['lr'],
        }
    
    def train(self, save_path: str = "checkpoints"):
        """Main training loop."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Reset environment
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle (obs, info) return
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        print(f"\n{'='*60}")
        print(f"Starting PPO Training: {self.task_name}")
        print(f"{'='*60}")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim: {self.action_dim}")
        print(f"Num envs: {self.num_envs}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for iteration in range(self.config.num_iterations):
            self.iteration = iteration
            iter_start = time.time()

            # Curriculum: hold place phase until reach success crosses threshold
            reach_only = False
            if self.task_name == "pick_place":
                if self.config.curriculum_place_start >= 0:
                    reach_only = iteration < self.config.curriculum_place_start
                else:
                    recent_reach = self.reach_successes[-self.config.curriculum_reach_window :] if self.reach_successes else []
                    mean_recent_reach = np.mean(recent_reach) if recent_reach else 0.0
                    reach_only = mean_recent_reach < self.config.curriculum_reach_threshold
                try:
                    # Vectorized env setter (pick_place only)
                    self.env.set_attr("reach_only", reach_only)
                except Exception:
                    pass
            
            # Collect rollout
            obs = self.collect_rollout(obs)
            
            # Update policy
            metrics = self.update()
            
            # Compute episode statistics
            if not hasattr(self, "episode_rewards"):
                self.episode_rewards = []
            if not hasattr(self, "episode_lengths"):
                self.episode_lengths = []
            if not hasattr(self, "episode_successes"):
                self.episode_successes = []
            mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            mean_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
            mean_success = np.mean(self.episode_successes[-100:]) if self.episode_successes else 0
            mean_reach_success = np.mean(self.reach_successes[-1000:]) if self.reach_successes else 0
            mean_place_success = np.mean(self.place_successes[-1000:]) if self.place_successes else 0
            mean_reach_dist = np.mean(self.reach_distances[-1000:]) if self.reach_distances else 0
            mean_place_dist = np.mean(self.place_distances[-1000:]) if self.place_distances else 0
            bc_buffer = len(self.success_replay)
            
            # Logging
            if iteration % self.config.log_interval == 0:
                fps = self.config.num_steps_per_env * self.num_envs / (time.time() - iter_start + 1e-6)
                elapsed = time.time() - start_time
                
                # Get current phase for pick_place
                current_phase = getattr(self, '_current_phase', 'REACH')
                phase_emoji = "🎯" if current_phase == "REACH" else "📦"
                
                # Calculate improvement/revert rates
                improvements = getattr(self, '_improvements', [])
                reverts = getattr(self, '_reverts', [])
                improve_rate = np.mean(improvements[-500:]) if improvements else 0
                revert_rate = np.mean(reverts[-500:]) if reverts else 0
                
                # Best distances
                best_reach = min(self.reach_distances) if self.reach_distances else float('inf')
                best_place = min(self.place_distances) if self.place_distances else float('inf')
                
                # EPISODE-LEVEL success rates (last 100 episodes)
                ep_reach_list = getattr(self, '_episode_reach_successes', [])
                ep_seq_list = getattr(self, '_episode_sequential_successes', [])
                
                ep_reach_rate = np.mean(ep_reach_list[-100:]) if ep_reach_list else 0.0
                ep_seq_rate = np.mean(ep_seq_list[-100:]) if ep_seq_list else 0.0
                
                print(f"\n{'='*110}")
                print(f"Iter {iteration:5d} | Phase: {phase_emoji} {current_phase} | Time: {elapsed:.0f}s | FPS: {fps:.0f}")
                print(f"{'='*110}")
                # Step-level progress (instantaneous)
                print(f"  PROGRESS:   Reached {mean_reach_success:6.2%} → Placed {mean_place_success:6.2%}")
                # Episode-level success rates (what really matters)
                print(f"  EPISODES:   Reach {ep_reach_rate:6.2%} → Sequential (reach+place) {ep_seq_rate:6.2%}  (last 100 eps)")
                print(f"  DISTANCES:  To cube: {mean_reach_dist:.4f}m (best: {best_reach:.4f}) | To target: {mean_place_dist:.4f}m (best: {best_place:.4f})")
                # Use step-level rewards for immediate feedback (episode rewards take too long)
                step_rewards = getattr(self, '_step_rewards', [])
                mean_step_reward = np.mean(step_rewards[-1000:]) if step_rewards else 0
                print(f"  REWARD:     Step avg: {mean_step_reward:8.2f} | Entropy: {metrics['entropy']:.4f} | Total episodes: {self._total_episodes}")
                print(f"  TRAINING:   Policy: {metrics['policy_loss']:.4f} | Value: {metrics['value_loss']:.4f} | LR: {metrics['lr']:.2e}")
                print(f"{'='*110}")
                
                if self.use_wandb:
                    wandb.log({
                        "iteration": iteration,
                        "total_steps": self.total_steps,
                        "reward/mean": mean_reward,
                        "reward/success_rate": mean_success,
                        "reward/reach_success_rate": mean_reach_success,
                        "reward/place_success_rate": mean_place_success,
                        "reward/reach_distance_mean": mean_reach_dist,
                        "reward/place_distance_mean": mean_place_dist,
                        "reward/episode_length": mean_length,
                        "loss/policy": metrics["policy_loss"],
                        "loss/value": metrics["value_loss"],
                        "loss/entropy": metrics["entropy"],
                        "loss/kl": metrics["kl"],
                        "loss/bc": metrics["bc_loss"],
                        "bc/buffer_size": bc_buffer,
                        "training/lr": metrics["lr"],
                        "training/fps": fps,
                    })
            
            # Save checkpoint
            if iteration % self.config.save_interval == 0 and iteration > 0:
                self.save(f"{save_path}/model_{iteration}.pt")
        
        # Final save
        self.save(f"{save_path}/model_final.pt")
        
        if self.use_wandb:
            wandb.finish()

        # Close viewer if opened
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
        
        print(f"\nTraining complete! Total time: {time.time() - start_time:.0f}s")
        
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "total_steps": self.total_steps,
            "config": self.config,
        }, path)
        print(f"Saved checkpoint: {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.iteration = checkpoint.get("iteration", 0)
        self.total_steps = checkpoint.get("total_steps", 0)
        print(f"Loaded checkpoint from: {path}")
        print(f"  Resuming from iteration {self.iteration}, total steps {self.total_steps}")

