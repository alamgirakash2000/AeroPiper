#!/usr/bin/env python3
"""
Pick-and-Place Training Script with Curriculum Learning
========================================================

Trains a policy for the pick-and-place task with:
- Curriculum learning (reach first, then place)
- Gradual progress tracking (memory of best control points)
- Detailed progress metrics

Usage:
    python scripts/train_pick_place.py --num-envs 4 --iterations 10000 --lr 2e-4 --device cuda
    python scripts/train_pick_place.py --render-window --render-interval 1
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from rl.ppo import PPO, PPOConfig
from rl.vec_env import VecEnv


def create_pick_place_env(
    max_episode_steps: int = 2000,
    success_threshold: float = 0.10,
    action_scale: float = 0.3,
    frame_skip: int = 5,
):
    """Create pick-and-place environment with curriculum support."""
    from envs.wrappers.pick_place_curriculum import AeroPiperPickPlaceCurriculum
    
    return AeroPiperPickPlaceCurriculum(
        max_episode_steps=max_episode_steps,
        reward_type="dense",
        success_threshold=success_threshold,
        randomize_objects=False,  # Start fixed, unlock via curriculum
        action_scale=action_scale,
        frame_skip=frame_skip,
        distance_tolerance=0.002,
        revert_on_regression=True,
        phase_transition_threshold=0.80,
        randomize_after_streak=5,
    )


class PickPlaceProgressTracker:
    """Tracks and displays training progress for pick-and-place."""
    
    def __init__(self):
        self.reach_successes = []
        self.place_successes = []
        self.reach_distances = []
        self.place_distances = []
        self.improvements = []
        self.reverts = []
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Best ever
        self.best_reach_dist = float('inf')
        self.best_place_dist = float('inf')
        
        # Current phase
        self.current_phase = "REACH"
        
    def update(self, info_list):
        """Update from step info."""
        for info in info_list:
            if not isinstance(info, dict):
                continue
            
            # Track successes
            if "success_reach" in info:
                self.reach_successes.append(float(info["success_reach"]))
            if "success_place" in info:
                self.place_successes.append(float(info["success_place"]))
            
            # Track distances
            if "dist_cube_hand" in info:
                self.reach_distances.append(float(info["dist_cube_hand"]))
                if info["dist_cube_hand"] < self.best_reach_dist:
                    self.best_reach_dist = info["dist_cube_hand"]
            if "dist_cube_target" in info:
                self.place_distances.append(float(info["dist_cube_target"]))
                if info["dist_cube_target"] < self.best_place_dist:
                    self.best_place_dist = info["dist_cube_target"]
            
            # Track improvements/reverts
            if "reach_improved" in info:
                self.improvements.append(float(info["reach_improved"]))
            if "reach_reverted" in info:
                self.reverts.append(float(info["reach_reverted"]))
            
            # Track phase
            if "curriculum_phase" in info:
                self.current_phase = info["curriculum_phase"]
            
            # Episode stats
            if "episode_reward" in info:
                self.episode_rewards.append(info["episode_reward"])
            if "episode_length" in info:
                self.episode_lengths.append(info["episode_length"])
    
    def get_metrics(self, window: int = 500) -> dict:
        """Get recent metrics."""
        return {
            "reach_success_rate": np.mean(self.reach_successes[-window:]) if self.reach_successes else 0,
            "place_success_rate": np.mean(self.place_successes[-window:]) if self.place_successes else 0,
            "avg_reach_dist": np.mean(self.reach_distances[-window:]) if self.reach_distances else float('inf'),
            "avg_place_dist": np.mean(self.place_distances[-window:]) if self.place_distances else float('inf'),
            "best_reach_dist": self.best_reach_dist,
            "best_place_dist": self.best_place_dist,
            "improvement_rate": np.mean(self.improvements[-window:]) if self.improvements else 0,
            "revert_rate": np.mean(self.reverts[-window:]) if self.reverts else 0,
            "current_phase": self.current_phase,
            "total_episodes": len(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
        }
    
    def format_progress(self, iteration: int, ppo_metrics: dict) -> str:
        """Format progress for printing."""
        m = self.get_metrics()
        
        phase_marker = "🎯 REACH" if m["current_phase"] == "REACH" else "📦 PLACE"
        
        lines = [
            "",
            f"{'='*90}",
            f"  Iter {iteration:6d} | Phase: {phase_marker} | Episodes: {m['total_episodes']:,}",
            f"{'='*90}",
            f"  SUCCESS RATES:",
            f"    Reach: {m['reach_success_rate']:6.2%} | Place: {m['place_success_rate']:6.2%}",
            f"  DISTANCES:",
            f"    Reach (avg/best): {m['avg_reach_dist']:.4f} / {m['best_reach_dist']:.4f} m",
            f"    Place (avg/best): {m['avg_place_dist']:.4f} / {m['best_place_dist']:.4f} m",
            f"  LEARNING:",
            f"    Improvement rate: {m['improvement_rate']:6.2%} | Revert rate: {m['revert_rate']:6.2%}",
            f"    Avg Reward: {m['avg_reward']:8.2f}",
            f"  PPO:",
            f"    Policy Loss: {ppo_metrics.get('policy_loss', 0):.4f} | "
            f"Value Loss: {ppo_metrics.get('value_loss', 0):.4f} | "
            f"LR: {ppo_metrics.get('lr', 0):.2e}",
            f"{'='*90}",
        ]
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Train Pick-and-Place with Curriculum Learning")
    
    # Training parameters
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--iterations", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--steps-per-env", type=int, default=128, help="Steps per env per iteration")
    
    # Environment parameters
    parser.add_argument("--max-episode-steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--success-threshold", type=float, default=0.10, help="Success distance threshold (m)")
    parser.add_argument("--action-scale", type=float, default=0.3, help="Action scale for gradual movement")
    
    # Rendering
    parser.add_argument("--render-window", action="store_true", help="Open viewer window")
    parser.add_argument("--render-interval", type=int, default=1, help="Steps between renders")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=10, help="Iterations between logs")
    parser.add_argument("--save-interval", type=int, default=500, help="Iterations between saves")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="aeropiper-pick-place", help="W&B project")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="checkpoints/pick_place_curriculum", help="Save directory")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("\n" + "="*60)
    print("Pick-and-Place Training with Curriculum Learning")
    print("="*60)
    print(f"  Num Envs: {args.num_envs}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Device: {args.device}")
    print(f"  Action Space: 6D (arm_select + 5 joints, joint6=0)")
    print("="*60 + "\n")
    
    # Create save directory
    save_dir = Path(args.save_dir) / time.strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {save_dir}")
    
    # Create environment factory
    def env_fn():
        return create_pick_place_env(
            max_episode_steps=args.max_episode_steps,
            success_threshold=args.success_threshold,
            action_scale=args.action_scale,
        )
    
    # Create vectorized environment
    print(f"\nCreating {args.num_envs} parallel environments...")
    vec_env = VecEnv(env_fn, num_envs=args.num_envs, device=args.device)
    
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")
    
    # Create PPO config
    ppo_config = PPOConfig(
        num_iterations=args.iterations,
        num_steps_per_env=args.steps_per_env,
        num_envs=args.num_envs,
        learning_rate=args.lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.005,  # Lower entropy for more focused exploration
        max_grad_norm=1.0,
        num_mini_batches=4,
        num_epochs=5,
        hidden_dims=[256, 256, 128],
        activation="elu",
        init_noise_std=0.5,  # Lower initial noise for more gradual actions
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        device=args.device,
    )
    
    # Create PPO trainer
    ppo = PPO(
        env=vec_env,
        config=ppo_config,
        task_name="pick_place_curriculum",
        reward_fn=None,  # Using curriculum wrapper's reward
        wandb_project=args.wandb_project if args.wandb else None,
        render_window=args.render_window,
        render_interval=args.render_interval,
    )
    
    # Resume if specified
    if args.resume:
        ppo.load(args.resume)
    
    # Create progress tracker
    tracker = PickPlaceProgressTracker()
    
    # Custom training loop with progress tracking
    print("\nStarting training...")
    print("Phase 1: Learning to REACH the cube")
    print("Phase 2: Learning to PLACE at target (unlocks at 80% reach success)")
    print("-" * 60)
    
    # Reset environment
    obs = vec_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32, device=args.device)
    
    start_time = time.time()
    
    for iteration in range(args.iterations):
        iter_start = time.time()
        
        # Collect rollout
        obs = ppo.collect_rollout(obs)
        
        # Track progress from collected info
        # Note: PPO tracks some info internally, we need to access it
        
        # Update policy
        metrics = ppo.update()
        
        # Get and track info from environments
        # The info is tracked during collect_rollout
        tracker.update(ppo.buffer.dones.cpu().numpy().tolist())  # Simplified tracking
        
        # Update tracker from PPO's tracked metrics
        tracker.reach_successes.extend([0.0] * ppo.config.num_steps_per_env * args.num_envs)
        tracker.place_successes.extend([0.0] * ppo.config.num_steps_per_env * args.num_envs)
        
        # Use PPO's internal tracking
        if hasattr(ppo, 'reach_successes') and ppo.reach_successes:
            tracker.reach_successes = ppo.reach_successes.copy()
        if hasattr(ppo, 'place_successes') and ppo.place_successes:
            tracker.place_successes = ppo.place_successes.copy()
        if hasattr(ppo, 'reach_distances') and ppo.reach_distances:
            tracker.reach_distances = ppo.reach_distances.copy()
        if hasattr(ppo, 'place_distances') and ppo.place_distances:
            tracker.place_distances = ppo.place_distances.copy()
        
        # Logging
        if iteration % args.log_interval == 0:
            fps = ppo.config.num_steps_per_env * args.num_envs / (time.time() - iter_start + 1e-6)
            elapsed = time.time() - start_time
            
            # Get metrics from PPO
            mean_reach = np.mean(ppo.reach_successes[-500:]) if hasattr(ppo, 'reach_successes') and ppo.reach_successes else 0
            mean_place = np.mean(ppo.place_successes[-500:]) if hasattr(ppo, 'place_successes') and ppo.place_successes else 0
            mean_reach_dist = np.mean(ppo.reach_distances[-500:]) if hasattr(ppo, 'reach_distances') and ppo.reach_distances else float('inf')
            mean_place_dist = np.mean(ppo.place_distances[-500:]) if hasattr(ppo, 'place_distances') and ppo.place_distances else float('inf')
            
            # Determine phase
            phase = "REACH" if mean_reach < 0.80 else "PLACE"
            phase_emoji = "🎯" if phase == "REACH" else "📦"
            
            print(f"\n{'='*90}")
            print(f"Iter {iteration:6d} | Phase: {phase_emoji} {phase} | Time: {elapsed:.0f}s | FPS: {fps:.0f}")
            print(f"{'='*90}")
            print(f"  SUCCESS RATES:  Reach: {mean_reach:6.2%} | Place: {mean_place:6.2%}")
            print(f"  DISTANCES:      Reach: {mean_reach_dist:.4f} m | Place: {mean_place_dist:.4f} m")
            print(f"  TRAINING:       Policy Loss: {metrics['policy_loss']:.4f} | Value Loss: {metrics['value_loss']:.4f}")
            print(f"{'='*90}")
        
        # Save checkpoint
        if iteration % args.save_interval == 0 and iteration > 0:
            ppo.save(str(save_dir / f"model_{iteration}.pt"))
    
    # Final save
    ppo.save(str(save_dir / "model_final.pt"))
    
    # Cleanup
    vec_env.close()
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Total Time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Checkpoints saved to: {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

