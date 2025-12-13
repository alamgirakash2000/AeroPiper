#!/usr/bin/env python3
"""
Generalized Evaluation Script
=============================

Evaluate trained RL policies with visualization.

Usage:
    python scripts/eval.py --task pick_place --checkpoint checkpoints/pick_place/.../model_final.pt
    python scripts/eval.py --task pick_place --checkpoint path/to/model.pt --episodes 10
    python scripts/eval.py --task pick_place --checkpoint path/to/model.pt --no-render
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import mujoco
import mujoco.viewer

from rl.networks import ActorCritic
from rl.configs import PickPlaceConfig, AssemblyConfig, HandoverConfig


def get_env(task: str, config, render_mode: str = None):
    """Get environment for the specified task."""
    if task == "pick_place":
        # Pick-place uses curriculum wrapper (6D action: arm_selector + 5 joints)
        from envs.wrappers.pick_place_curriculum import AeroPiperPickPlaceCurriculum
        return AeroPiperPickPlaceCurriculum(
            max_episode_steps=config.max_episode_steps,
            reward_type=config.reward_type,
            success_threshold=config.success_threshold,
            randomize_objects=config.randomize_objects,
            action_scale=config.action_scale,
            frame_skip=config.frame_skip,
            render_mode=render_mode,
        )
    elif task == "assembly":
        from envs.wrappers.assembly_wrapper import AeroPiperAssemblyEnv
        return AeroPiperAssemblyEnv(
            max_episode_steps=config.max_episode_steps,
            render_mode=render_mode,
        )
    elif task == "handover":
        from envs.wrappers.handover_wrapper import AeroPiperHandoverEnv
        return AeroPiperHandoverEnv(
            max_episode_steps=config.max_episode_steps,
            render_mode=render_mode,
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def get_task_config(task: str):
    """Get default configuration for the specified task."""
    configs = {
        "pick_place": PickPlaceConfig,
        "assembly": AssemblyConfig,
        "handover": HandoverConfig,
    }
    if task not in configs:
        raise ValueError(f"Unknown task: {task}")
    return configs[task]()


def load_policy(checkpoint_path: str, obs_dim: int, action_dim: int, device: str):
    """Load trained policy from checkpoint."""
    # PyTorch 2.6 defaults to weights_only=True, which prevents loading
    # checkpoints that store config dataclasses (e.g., rl.ppo.PPOConfig).
    # These checkpoints are produced by this repo and are trusted, so we
    # explicitly allow full pickle loading here for backward compatibility.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    
    # Get config from checkpoint or use defaults
    config = checkpoint.get("config", None)
    if config:
        hidden_dims = config.hidden_dims
        activation = config.activation
    else:
        hidden_dims = [256, 256, 256]
        activation = "elu"
    
    # Create policy
    policy = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        activation=activation,
    ).to(device)
    
    # Load weights
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    
    return policy


def evaluate_with_viewer(env, policy, num_episodes: int, device: str, max_steps: int = 500):
    """Evaluate policy with MuJoCo viewer visualization."""
    
    print("\n" + "="*60)
    print("Evaluation with Visualization")
    print("="*60)
    
    episode_rewards = []
    episode_successes = []
    episode_lengths = []
    
    # Get the actual MuJoCo model/data from nested wrapper
    base_env = env
    while hasattr(base_env, '_env'):
        base_env = base_env._env
    
    with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
        for episode in range(num_episodes):
            obs, info = env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            total_reward = 0
            success = False
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            for step in range(max_steps):
                # Get action from policy
                with torch.no_grad():
                    action, _, _ = policy.act(obs_tensor, deterministic=True)
                action = action.squeeze(0).cpu().numpy()
                action = np.clip(action, -1.0, 1.0)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                
                # Debug: print actions and distances AFTER step
                if step % 50 == 0:
                    reach_dist = info.get("dist_to_cube", 0)
                    place_dist = info.get("dist_palm_to_target", 0)
                    phase = info.get("phase", "?")
                    print(f"    Step {step}: action={action[:3]}..., reach={reach_dist:.3f}m, place={place_dist:.3f}m, phase={phase}")
                
                total_reward += reward
                
                # Check success
                if info.get("success", False):
                    success = True
                
                # Update viewer
                viewer.sync()
                time.sleep(0.02)  # Slow down for visualization
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
            episode_successes.append(success)
            episode_lengths.append(step + 1)
            
            print(f"  Reward: {total_reward:.2f}, Success: {success}, Steps: {step + 1}")
            
            # Brief pause between episodes
            time.sleep(0.5)
        
        print("\n" + "="*60)
        print("Evaluation Summary")
        print("="*60)
        print(f"Mean Reward:   {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Success Rate:  {np.mean(episode_successes)*100:.1f}%")
        print(f"Mean Length:   {np.mean(episode_lengths):.1f}")
        print("="*60)
        
        print("\nClose viewer to exit...")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.1)
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "success_rate": np.mean(episode_successes),
        "mean_length": np.mean(episode_lengths),
    }


def evaluate_headless(env, policy, num_episodes: int, device: str, max_steps: int = 500):
    """Evaluate policy without visualization (faster)."""
    
    print("\n" + "="*60)
    print("Evaluation (Headless)")
    print("="*60)
    
    episode_rewards = []
    episode_successes = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        total_reward = 0
        success = False
        
        for step in range(max_steps):
            with torch.no_grad():
                action, _, _ = policy.act(obs_tensor, deterministic=True)
            action = action.squeeze(0).cpu().numpy()
            action = np.clip(action, -1.0, 1.0)
            
            obs, reward, terminated, truncated, info = env.step(action)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            total_reward += reward
            if info.get("success", False):
                success = True
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_successes.append(success)
        episode_lengths.append(step + 1)
        
        print(f"Episode {episode + 1}: Reward={total_reward:.2f}, Success={success}, Steps={step + 1}")
    
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    print(f"Mean Reward:   {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Success Rate:  {np.mean(episode_successes)*100:.1f}%")
    print(f"Mean Length:   {np.mean(episode_lengths):.1f}")
    print("="*60)
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "success_rate": np.mean(episode_successes),
        "mean_length": np.mean(episode_lengths),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL policy")
    
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["pick_place", "assembly", "handover"],
        help="Task to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--no-render", action="store_true", help="Disable visualization")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get config
    config = get_task_config(args.task)
    
    # Create environment
    env = get_env(args.task, config)
    
    print(f"\nTask: {args.task}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Load policy
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy = load_policy(
        args.checkpoint,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device,
    )
    
    print(f"Device: {device}")
    print(f"Loaded policy from: {args.checkpoint}")
    
    # Evaluate
    if args.no_render:
        results = evaluate_headless(env, policy, args.episodes, device, args.max_steps)
    else:
        results = evaluate_with_viewer(env, policy, args.episodes, device, args.max_steps)
    
    env.close()
    
    return results


if __name__ == "__main__":
    main()

