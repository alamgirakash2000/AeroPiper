#!/usr/bin/env python3
"""
Generalized Training Script
===========================

Train RL policies for any AeroPiper task.

Usage:
    python scripts/train.py --task pick_place
    python scripts/train.py --task assembly --num-envs 1024
    python scripts/train.py --task handover --wandb
    
    # With custom config
    python scripts/train.py --task pick_place --lr 1e-4 --iterations 10000
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rl.ppo import PPO, PPOConfig
from rl.vec_env import VecEnv
from rl.configs import PickPlaceConfig, AssemblyConfig, HandoverConfig
from rl.rewards import PickPlaceReward, AssemblyReward, HandoverReward


def get_env_factory(task: str, config):
    """Get environment factory function for the specified task."""
    if task == "pick_place":
        # Pick-place uses curriculum wrapper with gradual learning
        # 6D action: [arm_select, j1, j2, j3, j4, j5], joint6=0
        from envs.wrappers.pick_place_curriculum import AeroPiperPickPlaceCurriculum
        
        # Check if curriculum should be disabled
        disable_curriculum = getattr(config, 'disable_curriculum', False)
        
        def env_fn():
            return AeroPiperPickPlaceCurriculum(
                max_episode_steps=config.max_episode_steps,
                reward_type=config.reward_type,
                success_threshold=config.success_threshold,
                randomize_objects=config.randomize_objects,
                action_scale=config.action_scale,
                frame_skip=config.frame_skip,
                distance_tolerance=0.002,  # 2mm tolerance for gradual progress
                revert_on_regression=True,  # Revert to best position on regression
                phase_transition_threshold=0.80,  # 80% reach success to unlock place
                randomize_after_streak=5,  # Randomize after 5 consecutive successes
                disable_curriculum=disable_curriculum,  # Train reach+place together
            )
        return env_fn
        
    elif task == "assembly":
        from envs.wrappers.assembly_wrapper import AeroPiperAssemblyEnv
        
        def env_fn():
            return AeroPiperAssemblyEnv(
                max_episode_steps=config.max_episode_steps,
                reward_type=config.reward_type,
                action_scale=config.action_scale,
                frame_skip=config.frame_skip,
            )
        return env_fn
        
    elif task == "handover":
        from envs.wrappers.handover_wrapper import AeroPiperHandoverEnv
        
        def env_fn():
            return AeroPiperHandoverEnv(
                max_episode_steps=config.max_episode_steps,
                reward_type=config.reward_type,
                action_scale=config.action_scale,
                frame_skip=config.frame_skip,
            )
        return env_fn
    
    else:
        raise ValueError(f"Unknown task: {task}")


def get_reward_fn(task: str, config):
    """Get reward function for the specified task."""
    if task == "pick_place":
        return PickPlaceReward()
    elif task == "assembly":
        return AssemblyReward()
    elif task == "handover":
        return HandoverReward()
    return None


def get_task_config(task: str):
    """Get default configuration for the specified task."""
    configs = {
        "pick_place": PickPlaceConfig,
        "assembly": AssemblyConfig,
        "handover": HandoverConfig,
    }
    if task not in configs:
        raise ValueError(f"Unknown task: {task}. Available: {list(configs.keys())}")
    return configs[task]()


def main():
    parser = argparse.ArgumentParser(description="Train RL policy for AeroPiper tasks")
    
    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["pick_place", "assembly", "handover"],
        help="Task to train on",
    )
    
    # Training parameters
    parser.add_argument("--num-envs", type=int, default=None, help="Number of parallel environments (defaults to config)")
    parser.add_argument("--iterations", type=int, default=None, help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint path")
    parser.add_argument("--max-episode-steps", type=int, default=None, help="Max steps per episode (default: 2000)")
    parser.add_argument("--randomize", action="store_true", help="Randomize object positions each episode")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum (learn reach first, then place)")
    
    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="aeropiper-rl", help="W&B project name")
    parser.add_argument("--run-name", type=str, default=None, help="W&B run name")
    
    # Rendering
    parser.add_argument(
        "--render-window",
        action="store_true",
        help="Open a Mujoco viewer window for env 0 during training",
    )
    parser.add_argument(
        "--render-interval",
        type=int,
        default=1,
        help="How many rollout steps between viewer refreshes (use larger to reduce overhead)",
    )
    
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Get task config
    config = get_task_config(args.task)
    
    # Override config with command line args
    if args.num_envs is not None:
        config.num_envs = args.num_envs
    if args.iterations is not None:
        config.num_iterations = args.iterations
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.device:
        config.device = args.device
    if args.max_episode_steps is not None:
        config.max_episode_steps = args.max_episode_steps
    if args.randomize:
        config.randomize_objects = True
        print("Randomization enabled: cube and target positions will vary each episode")
    if args.curriculum:
        config.disable_curriculum = False
        print("Curriculum enabled: learn reach first, then place")
    else:
        config.disable_curriculum = True  # Default: no curriculum
    
    # Print configuration
    print("\n" + "="*60)
    print(f"Training Configuration: {args.task}")
    print("="*60)
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Create save directory
    save_dir = Path(args.save_dir) / args.task / time.strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir}")
    
    # Create environment factory
    env_fn = get_env_factory(args.task, config)
    
    # Reset curriculum state for pick_place
    if args.task == "pick_place":
        from envs.wrappers.pick_place_curriculum import AeroPiperPickPlaceCurriculum
        AeroPiperPickPlaceCurriculum.reset_curriculum()
        print("Curriculum reset: Starting in REACH phase")
    
    # Create vectorized environment
    print(f"\nCreating {config.num_envs} parallel environments...")
    vec_env = VecEnv(env_fn, num_envs=config.num_envs, device=config.device)
    
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")
    
    # Mark training as started (for curriculum to start counting episodes)
    if args.task == "pick_place":
        AeroPiperPickPlaceCurriculum.start_training()
        print("Curriculum: Training started, episode counting enabled")
    
    # Create PPO config
    ppo_config = PPOConfig(
        num_iterations=config.num_iterations,
        num_steps_per_env=config.num_steps_per_env,
        num_envs=config.num_envs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_ratio=config.clip_ratio,
        value_loss_coef=config.value_loss_coef,
        entropy_coef=config.entropy_coef,
        max_grad_norm=config.max_grad_norm,
        learning_rate=config.learning_rate,
        lr_schedule=config.lr_schedule,
        num_mini_batches=config.num_mini_batches,
        num_epochs=config.num_epochs,
        hidden_dims=config.hidden_dims,
        activation=config.activation,
        init_noise_std=config.init_noise_std,
        log_interval=config.log_interval,
        save_interval=config.save_interval,
        device=config.device,
    )
    
    # Get reward function (optional shaping)
    reward_fn = get_reward_fn(args.task, config)
    
    # Create PPO trainer
    ppo = PPO(
        env=vec_env,
        config=ppo_config,
        task_name=args.task,
        reward_fn=reward_fn,
        wandb_project=args.wandb_project if args.wandb else None,
        wandb_run_name=args.run_name,
        render_window=args.render_window,
        render_interval=args.render_interval,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        ppo.load(args.resume)
    
    # Train
    ppo.train(save_path=str(save_dir))
    
    # Cleanup
    vec_env.close()
    
    print(f"\nTraining complete! Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()

