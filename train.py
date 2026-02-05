"""
Safe Stop Training Script - Enhanced Version

Features:
- Curriculum learning for gradual difficulty increase
- Better logging and visualization
- Checkpoint management
- Performance tracking
- Evaluation utilities
"""

import os
import sys
import argparse
from datetime import datetime
import torch
import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any
from collections import deque
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.safe_stop_env import (
    BaseEnv,
    BaseEnvConfig,
    ConservativeAvoidanceEnv,
    EnvRenderer,
    TerminationReason
)


class ObsDict(dict):
    """Dictionary wrapper that supports .to(device) for rsl_rl compatibility."""
    def to(self, device):
        result = ObsDict()
        for k, v in self.items():
            if hasattr(v, 'to'):
                result[k] = v.to(device)
            else:
                result[k] = v
        return result


class GymEnvWrapper:
    """Gym-style wrapper for BaseEnv subclasses."""

    def __init__(self, env: BaseEnv):
        self.env = env
        self.observation_space = type('Space', (), {
            'shape': (env.obs_dim,),
            'dtype': np.float32
        })()
        self.action_space = type('Space', (), {
            'shape': (env.action_dim,),
            'dtype': np.float32,
            'low': -1.0,
            'high': 1.0
        })()
        self._max_episode_steps = env.cfg.max_episode_steps

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        obs = self.env.reset(seed=seed)
        return obs, {"threat_active": self.env.threat_active}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if isinstance(action, np.ndarray):
            action = action[0] if action.ndim > 0 else float(action)

        obs, reward, done, info = self.env.step(action)  # info already has rich data!

        terminated = info.get('termination_reason', 'none') == 'collision'
        truncated = info.get('termination_reason', 'none') in ['timeout', 'success']

        # Don't overwrite info, just add to it
        info['speed'] = abs(self.env.ego_vy)
        info['is_still'] = float(abs(self.env.ego_vy) < self.env.cfg.stillness_threshold)

        return obs, reward, terminated, truncated, info


class VecEnvWrapper:
    """Vectorized environment wrapper compatible with rsl_rl OnPolicyRunner."""

    def __init__(
        self,
        env_fn: Callable,
        num_envs: int,
        device: str = "cuda:0",
        seed: int = 0,
        render: bool = False,
        render_fps: int = 8,
    ):
        self.device = device
        self.num_envs = num_envs
        self.render_enabled = render and num_envs == 1
        self.render_fps = render_fps
        self.renderer = None

        # Create environments
        self.envs = [env_fn() for _ in range(num_envs)]
        for i, env in enumerate(self.envs):
            env.reset(seed=seed + i)

        # Get properties from first env
        self._env = self.envs[0]
        self.cfg = self._env.env.cfg
        self.num_obs = self._env.observation_space.shape[0]
        self.num_actions = self._env.action_space.shape[0]
        self._max_episode_length = self._env._max_episode_steps

        # Tensor buffers
        self.obs_buf = torch.zeros((num_envs, self.num_obs), device=device, dtype=torch.float32)
        self.reward_buf = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.reset_buf = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.episode_return_buf = torch.zeros(num_envs, device=device, dtype=torch.float32)

        # Tracking
        self.episode_returns: deque = deque(maxlen=1000)
        self.episode_lengths: deque = deque(maxlen=1000)
        self.episode_collisions: deque = deque(maxlen=1000)
        self.episode_successes: deque = deque(maxlen=1000)
        self.episode_avg_speed: deque = deque(maxlen=1000)
        self.episode_stillness: deque = deque(maxlen=1000)

        self.extras: Dict[str, Any] = {}
        self.last_reward_components: Dict[str, float] = {}  # ADD THIS
        # Setup renderer
        if self.render_enabled:
            self.renderer = EnvRenderer(self.envs[0].env, scale=20.0)

        self._reset_all()

    @property
    def max_episode_length(self):
        return self._max_episode_length

    def _reset_all(self):
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            self.obs_buf[i] = torch.from_numpy(obs).to(self.device)
        self.episode_length_buf.zero_()
        self.episode_return_buf.zero_()

    def _reset_env(self, idx: int):
        obs, _ = self.envs[idx].reset()
        self.obs_buf[idx] = torch.from_numpy(obs).to(self.device)
        self.episode_length_buf[idx] = 0
        self.episode_return_buf[idx] = 0.0

    def get_observations(self) -> Dict:
        obs_dict = ObsDict()
        obs_dict["observations"] = self.obs_buf
        return obs_dict

    def reset(self) -> Dict:
        self._reset_all()
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> Tuple[Dict, torch.Tensor, torch.Tensor, Dict]:
        actions_np = actions.cpu().numpy()

        # Track reward components across all environments that finish episodes
        reward_components_sum = {}
        reward_components_count = 0

        for i, env in enumerate(self.envs):
            action_val = float(actions_np[i][0]) if actions_np[i].ndim > 0 else float(actions_np[i])
            obs, reward, terminated, truncated, info = env.step(actions_np[i])

            is_timeout = self.episode_length_buf[i] >= self.max_episode_length - 1
            done = terminated or truncated or is_timeout

            self.obs_buf[i] = torch.from_numpy(obs).to(self.device)
            self.reward_buf[i] = float(reward)
            self.reset_buf[i] = done
            self.episode_length_buf[i] += 1
            self.episode_return_buf[i] += reward

            if done:
                # Episode statistics
                self.episode_returns.append(self.episode_return_buf[i].item())
                self.episode_lengths.append(self.episode_length_buf[i].item())
                self.episode_collisions.append(float(info.get('collision', False)))
                term_reason = info.get('termination_reason', 'none')
                self.episode_successes.append(float(term_reason in ['timeout', 'success']))

                # Track average speed and stillness for this episode
                avg_speed = info.get('ego_speed', 0.0)
                self.episode_avg_speed.append(avg_speed)

                # Calculate stillness rate (approximate from final velocity)
                is_still = float(avg_speed < self.cfg.stillness_threshold)
                self.episode_stillness.append(is_still)

                # Aggregate reward components from this episode
                if 'reward_components' in info:
                    reward_components_count += 1
                    for key, value in info['reward_components'].items():
                        if key not in reward_components_sum:
                            reward_components_sum[key] = 0.0
                        reward_components_sum[key] += float(value)

                self._reset_env(i)

        # Store averaged reward components for logging
        if reward_components_count > 0:
            self.last_reward_components = {
                k: v / reward_components_count
                for k, v in reward_components_sum.items()
            }
        else:
            self.last_reward_components = {}

        self._update_extras()

        if self.render_enabled and self.renderer:
            self._render()

        obs_dict = ObsDict()
        obs_dict["observations"] = self.obs_buf
        return obs_dict, self.reward_buf, self.reset_buf, self.extras

    def _update_extras(self):
        self.extras = {
            'time_outs': self.reset_buf.clone(),
        }

        if self.episode_returns:
            ep_stats = {
                'episode_return': np.mean(self.episode_returns),
                'episode_length': np.mean(self.episode_lengths),
                'episode_collision_rate': np.mean(self.episode_collisions),
                'episode_success_rate': np.mean(self.episode_successes),
            }

            # Add speed and stillness metrics
            if self.episode_avg_speed:
                ep_stats['episode_avg_speed'] = np.mean(self.episode_avg_speed)
            if self.episode_stillness:
                ep_stats['episode_stillness_rate'] = np.mean(self.episode_stillness)

            # Add reward components to TensorBoard logging
            if hasattr(self, 'last_reward_components') and self.last_reward_components:
                for key, value in self.last_reward_components.items():
                    ep_stats[f'reward_component/{key}'] = float(value)

            self.extras['episode'] = ep_stats

    def _render(self):
        if self.renderer and self.num_envs == 1:
            env = self.envs[0].env
            reward = self.reward_buf[0].item() if len(self.reward_buf) > 0 else 0.0
            action = 0.0  # Could track this if needed
            self.renderer.render(reward=reward, action=action)
            self.renderer.tick(self.render_fps)

    def close(self):
        if self.renderer:
            self.renderer.close()


class CurriculumManager:
    """
    Manages curriculum learning by gradually increasing difficulty.
    """

    def __init__(self, config: BaseEnvConfig, total_timesteps: int):
        self.base_config = config
        self.total_timesteps = total_timesteps
        self.current_stage = 0

        # Define curriculum stages
        self.stages = [
            {
                'name': 'easy',
                'duration': 0.3,  # 30% of total training
                'obj_vx_range': (-1.0, -0.5),  # Slower objects
                'obj_spawn_x_range': (10.0, 15.0),  # Farther spawn
                'threat_threshold': 0.3,  # Larger threat margin
            },
            {
                'name': 'medium',
                'duration': 0.4,  # 40% of total training
                'obj_vx_range': (-1.5, -0.5),
                'obj_spawn_x_range': (8.0, 15.0),
                'threat_threshold': 0.25,
            },
            {
                'name': 'hard',
                'duration': 0.3,  # 30% of total training
                'obj_vx_range': (-2.0, -0.5),  # Full speed
                'obj_spawn_x_range': (6.0, 15.0),  # Full range
                'threat_threshold': 0.2,  # Original threshold
            },
        ]

    def update_config(self, current_timestep: int) -> BaseEnvConfig:
        """Update config based on training progress."""
        progress = current_timestep / self.total_timesteps

        # Determine current stage
        cumulative = 0.0
        for i, stage in enumerate(self.stages):
            cumulative += stage['duration']
            if progress <= cumulative:
                if i != self.current_stage:
                    self.current_stage = i
                    print(f"\n{'='*60}")
                    print(f"CURRICULUM: Advancing to stage '{stage['name']}'")
                    print(f"{'='*60}\n")

                # Apply stage parameters
                config = BaseEnvConfig(
                    obj_vx_range=stage['obj_vx_range'],
                    obj_spawn_x_range=stage['obj_spawn_x_range'],
                    threat_threshold=stage['threat_threshold'],
                )
                return config

        # If we're past all stages, use the hardest
        return BaseEnvConfig()


def make_env_fn(config: BaseEnvConfig):
    """Factory function to create environment instances."""
    def _init():
        env = ConservativeAvoidanceEnv(config)
        return GymEnvWrapper(env)
    return _init


def train(args):
    """Main training function."""
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError:
        print("Error: rsl_rl not installed. Install with: pip install rsl_rl")
        sys.exit(1)

    print("=" * 60)
    print("SAFE STOP TRAINING")
    print("=" * 60)

    device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Save args
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Environment config
    config = BaseEnvConfig(max_episode_steps=args.max_episode_steps)

    # Curriculum learning
    curriculum = None
    if args.use_curriculum:
        curriculum = CurriculumManager(config, args.total_timesteps)
        print("Curriculum learning enabled")

    # Create vectorized environment
    env = VecEnvWrapper(
        env_fn=make_env_fn(config),
        num_envs=args.n_envs,
        device=device,
        seed=args.seed,
        render=args.render,
        render_fps=8,
    )

    print(f"Obs dim: {env.num_obs}, Action dim: {env.num_actions}")
    print(f"Num envs: {args.n_envs}, Max ep len: {env.max_episode_length}")

    # Training config
    train_cfg = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": args.clip_range,
            "gamma": args.gamma,
            "lam": args.gae_lambda,
            "learning_rate": args.learning_rate,
            "max_grad_norm": args.max_grad_norm,
            "num_learning_epochs": args.n_epochs,
            "num_mini_batches": args.num_mini_batches,
            "value_loss_coef": args.vf_coef,
            "entropy_coef": args.ent_coef,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": args.desired_kl,
        },
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": args.hidden_dims,
            "critic_hidden_dims": args.hidden_dims,
            "activation": "elu",
            "init_noise_std": args.init_noise_std,
        },
        "num_steps_per_env": args.n_steps,
        "save_interval": args.save_interval,
        "empirical_normalization": args.normalize_obs,
        "logger": "tensorboard",
        "obs_groups": {
            "policy": ["observations"],
            "critic": ["observations"]
        },

    }

    runner = OnPolicyRunner(env=env, train_cfg=train_cfg, log_dir=log_dir, device=device)

    if args.resume:
        print(f"Resuming from: {args.resume}")
        runner.load(args.resume)

    steps_per_iter = args.n_steps * args.n_envs
    num_iterations = args.total_timesteps // steps_per_iter

    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Iterations: {num_iterations:,}")
    print(f"Steps per iteration: {steps_per_iter:,}")
    print("=" * 60)

    # Training loop with curriculum updates
    if curriculum:
        for iteration in range(num_iterations):
            current_timestep = iteration * steps_per_iter
            new_config = curriculum.update_config(current_timestep)

            # Update environment configs if needed
            # Note: This is a simplified approach; ideally we'd recreate envs

            runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
    else:
        runner.learn(num_learning_iterations=num_iterations, init_at_random_ep_len=True)

    final_path = os.path.join(log_dir, "final_model.pt")
    runner.save(final_path)
    print(f"\nSaved: {final_path}")

    env.close()

    return log_dir


def evaluate(args):
    """Evaluate trained model."""
    try:
        from rsl_rl.modules import ActorCritic
    except ImportError:
        print("Error: rsl_rl not installed")
        sys.exit(1)

    print("=" * 60)
    print("SAFE STOP EVALUATION")
    print("=" * 60)

    device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Environment
    config = BaseEnvConfig(max_episode_steps=args.max_episode_steps)
    env = ConservativeAvoidanceEnv(config)

    renderer = EnvRenderer(env, scale=20.0) if args.render else None

    # Load model
    print(f"Loading: {args.model_path}")

    # Create a dummy observation dict with the right shape
    dummy_obs = {"observations": torch.zeros(1, env.obs_dim, device=device)}

    obs_groups = {
        "policy": ["observations"],
        "critic": ["observations"]
    }

    policy = ActorCritic(
        num_actor_obs=env.obs_dim,
        num_critic_obs=env.obs_dim,
        num_actions=env.action_dim,
        actor_hidden_dims=args.hidden_dims,
        critic_hidden_dims=args.hidden_dims,
        activation="elu",
        init_noise_std=0.1,
        obs=dummy_obs,
        obs_groups=obs_groups,
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    print("Model loaded.")

    # Run episodes
    episode_returns = []
    episode_lengths = []
    episode_speeds = []
    episode_stillness = []
    collisions = 0
    successes = 0

    try:
        for ep in range(args.num_episodes):
            obs = env.reset()
            total_reward = 0.0
            steps = 0
            speeds = []
            still_count = 0
            done = False

            while not done:
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                obs_dict = {"observations": obs_tensor}  # <-- ADD THIS
                with torch.no_grad():
                    action = policy.act_inference(obs_dict)
                action_val = action.cpu().numpy().flatten()[0]

                obs, reward, done, info = env.step(action_val)
                total_reward += reward
                steps += 1

                # Track metrics
                speed = abs(env.ego_vy)
                speeds.append(speed)
                if speed < env.cfg.stillness_threshold:
                    still_count += 1

                if renderer:
                    import pygame
                    renderer.render(reward=reward, action=action_val)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (
                            event.type == pygame.KEYDOWN and
                            event.key in [pygame.K_q, pygame.K_ESCAPE]
                        ):
                            renderer.close()
                            return
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                            obs = env.reset()
                            total_reward = 0.0
                            steps = 0
                            speeds = []
                            still_count = 0
                    renderer.tick(8)

            episode_returns.append(total_reward)
            episode_lengths.append(steps)
            episode_speeds.append(np.mean(speeds) if speeds else 0.0)
            episode_stillness.append(still_count / steps if steps > 0 else 0.0)

            if info.get('collision', False):
                collisions += 1
            elif info.get('termination_reason') in ['timeout', 'success']:
                successes += 1

            print(f"Ep {ep+1}/{args.num_episodes}: "
                  f"reward={total_reward:.2f}, steps={steps}, "
                  f"avg_speed={episode_speeds[-1]:.3f}, "
                  f"stillness={episode_stillness[-1]*100:.1f}%, "
                  f"reason={info.get('termination_reason', 'none')}")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        if renderer:
            renderer.close()

    if episode_returns:
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Episodes: {len(episode_returns)}")
        print(f"Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
        print(f"Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"Avg Speed: {np.mean(episode_speeds):.3f} ± {np.std(episode_speeds):.3f} m/s")
        print(f"Stillness: {np.mean(episode_stillness)*100:.1f}% ± {np.std(episode_stillness)*100:.1f}%")
        print(f"Collision Rate: {collisions/len(episode_returns)*100:.1f}%")
        print(f"Success Rate: {successes/len(episode_returns)*100:.1f}%")
        print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safe Stop Training/Evaluation")

    # Mode
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])

    # Environment
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--n_envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    # Training
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    parser.add_argument("--n_steps", type=int, default=24)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--num_mini_batches", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--desired_kl", type=float, default=0.01)

    # Policy
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--init_noise_std", type=float, default=0.3)
    parser.add_argument("--normalize_obs", action="store_true")

    # Curriculum
    parser.add_argument("--use_curriculum", action="store_true",
                       help="Enable curriculum learning")

    # Logging
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None)

    # Evaluation
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")

    # Device
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        if args.model_path is None:
            print("Error: --model_path required for eval mode")
            sys.exit(1)
        evaluate(args)