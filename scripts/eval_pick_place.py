#!/usr/bin/env python3
"""
Evaluation script for pick-and-place policy.
Uses the exact same environment as training.
"""

import argparse
import numpy as np
import torch
import mujoco
import mujoco.viewer
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.wrappers.pick_place_curriculum import AeroPiperPickPlaceCurriculum
from rl.networks import ActorCritic
from rl.configs import PickPlaceConfig


def load_policy(checkpoint_path: str, obs_dim: int, action_dim: int, device: str):
    """Load trained policy from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint or use defaults
    config = checkpoint.get("config", None)
    if config is not None and hasattr(config, "hidden_dims"):
        hidden_dims = config.hidden_dims
        activation = getattr(config, "activation", "elu")
    else:
        hidden_dims = [256, 256, 128]
        activation = "elu"
    
    policy = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        activation=activation,
    ).to(device)
    
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    
    return policy


def main():
    parser = argparse.ArgumentParser(description="Evaluate pick-place policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--no-viewer", action="store_true", help="Run without viewer")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions (default: stochastic)")
    parser.add_argument("--fast", action="store_true", help="Fast mode without smooth movement (default: realtime)")
    parser.add_argument("--randomize", action="store_true", help="Randomize object positions each episode")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Create environment with same config as training
    config = PickPlaceConfig()
    
    # Realtime mode is default (use --fast to disable)
    realtime = not args.fast
    # For realtime mode, use frame_skip=1 for smooth visualization
    frame_skip = 1 if realtime else config.frame_skip
    # Adjust action_scale for realtime mode (smaller steps = smoother)
    action_scale = config.action_scale / 5 if realtime else config.action_scale
    
    env = AeroPiperPickPlaceCurriculum(
        max_episode_steps=args.max_steps,  # Use command line arg
        reward_type=config.reward_type,
        success_threshold=config.success_threshold,
        randomize_objects=args.randomize,  # Randomize positions if flag set
        action_scale=action_scale,
        frame_skip=frame_skip,
        render_mode=None,
    )
    
    if args.randomize:
        print("Randomization enabled: different positions each episode")
    
    if realtime:
        print(f"Realtime mode: frame_skip=1, action_scale={action_scale:.3f}")
    
    # Stochastic is default (use --deterministic to disable)
    stochastic = not args.deterministic
    print(f"Action mode: {'stochastic' if stochastic else 'deterministic'}")
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Load policy
    policy = load_policy(args.checkpoint, obs_dim, action_dim, device)
    print(f"Policy loaded successfully!")
    
    # Get base MuJoCo model/data for viewer
    base_env = env._env  # AeroPiperPickPlace
    model = base_env.model
    data = base_env.data
    
    print(f"\n{'='*70}")
    print("Starting Evaluation")
    print(f"{'='*70}")
    
    episode_rewards = []
    episode_successes = []
    
    # Create viewer context if needed
    viewer = None
    if not args.no_viewer:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90
    
    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            total_reward = 0
            reach_achieved = False
            place_achieved = False
            terminated = False
            truncated = False
            
            print(f"\n--- Episode {ep+1}/{args.episodes} ---")
            
            for step in range(args.max_steps):
                # Get action from policy
                with torch.no_grad():
                    action, _, _ = policy.act(obs_tensor, deterministic=not stochastic)
                action_np = action.squeeze(0).cpu().numpy()
                action_np = np.clip(action_np, -1.0, 1.0)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action_np)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                
                total_reward += reward
                
                # Track success
                if info.get("success_reach", False):
                    reach_achieved = True
                if info.get("success_place", False):
                    place_achieved = True
                
                # Sync viewer if enabled
                if viewer is not None:
                    viewer.sync()
                    if realtime:
                        # Real-time mode: simulate real robot speed
                        # With frame_skip=1, each step is 0.002s simulation time
                        # Sleep 0.02s for 10x slower than realtime (visible movement)
                        time.sleep(0.02)
                
                # Print progress periodically
                if step % 50 == 0:
                    reach_dist = info.get("dist_cube_hand", -1)
                    place_dist = info.get("dist_palm_target", -1)
                    phase = info.get("curriculum_phase", "?")
                    arm = info.get("active_arm", "R" if action_np[0] >= 0 else "L")
                    ep_reach = info.get("episode_reach_success", False)
                    ep_place = info.get("episode_place_success", False)
                    print(f"  Step {step:4d}: arm={arm}, reach={reach_dist:.3f}m, place={place_dist:.3f}m, phase={phase}, ep_reach={ep_reach}, rew={reward:.2f}")
                
                if terminated or truncated:
                    print(f"  Episode ended: terminated={terminated}, truncated={truncated}")
                    break
            else:
                # Loop completed without break = hit max_steps = truncated
                truncated = True
                print(f"  Episode ended: terminated=False, truncated=True (hit max_steps)")
            
            # Success only if task completed without truncation (within time limit)
            # Truncated = hit max steps = task not completed in time
            success = reach_achieved and place_achieved and not truncated
            episode_rewards.append(total_reward)
            episode_successes.append(success)
            
            status = "✅ SUCCESS" if success else ("⏱️ TIMEOUT" if truncated else "❌ FAILED")
            print(f"  RESULT: {status} | reward={total_reward:.2f}, reach={reach_achieved}, place={place_achieved}")
            
            if viewer is not None:
                time.sleep(0.5)
        
        print(f"\n{'='*70}")
        print("Evaluation Summary")
        print(f"{'='*70}")
        print(f"Mean Reward:  {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Success Rate: {np.mean(episode_successes)*100:.1f}%")
        print(f"{'='*70}")
        
        if viewer is not None:
            print("\nClose viewer to exit...")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.1)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()

