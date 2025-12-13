"""
Neural Network Architectures for RL
===================================
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Args:
        obs_dim: Observation space dimension
        action_dim: Action space dimension
        hidden_dims: List of hidden layer sizes
        activation: Activation function
        init_noise_std: Initial standard deviation for action distribution
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Activation function
        activation_fn = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
        }[activation]
        
        # Build actor network
        actor_layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
            ])
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # Build critic network
        critic_layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
            ])
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Action noise (learnable log std)
        self.log_std = nn.Parameter(torch.ones(action_dim) * torch.log(torch.tensor(init_noise_std)))
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor):
        """Forward pass returning action mean and value."""
        action_mean = self.actor(obs)
        value = self.critic(obs)
        return action_mean, value
    
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Sample action from policy.
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: Value estimate
        """
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = Normal(action_mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate actions for PPO update.
        
        Returns:
            log_prob: Log probability of actions
            value: Value estimate
            entropy: Entropy of action distribution
        """
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        
        dist = Normal(action_mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value.squeeze(-1), entropy
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        return self.critic(obs).squeeze(-1)

