import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from envs.safe_stop_env import SafeStopEnv, BaseEnvConfig

class CurriculumCallback(BaseCallback):
    """
    Gradually increases difficulty by modifying the environment config 
    across parallel environments.
    """
    def __init__(self, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        
        # Access the underlying environments in the VecEnv
        envs = self.training_env.envs
        
        # Adjust difficulty based on progress
        if progress < 0.3:
            vx_range = (-1.0, -0.5) # Easy
        elif progress < 0.7:
            vx_range = (-1.5, -0.5) # Medium
        else:
            vx_range = (-2.0, -0.5) # Hard
            
        for env in envs:
            env.unwrapped.cfg.obj_vx_range = vx_range
            
        return True

def main(args):
    log_dir = "./logs/safe_stop_ppo"
    os.makedirs(log_dir, exist_ok=True)

    # 1. Create vectorized environment using standard SB3 utility
    env = make_vec_env(SafeStopEnv, n_envs=args.n_envs)
    eval_env = make_vec_env(SafeStopEnv, n_envs=1)

    # 2. Setup Callbacks
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=10000,
                                 deterministic=True, render=False)
    curriculum_callback = CurriculumCallback(total_timesteps=args.total_timesteps)

    # 3. Initialize PPO
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
                learning_rate=3e-4, batch_size=256, ent_coef=0.01)

    # 4. Train
    print("Starting training...")
    model.learn(total_timesteps=args.total_timesteps, 
                callback=[eval_callback, curriculum_callback])
    
    model.save(f"{log_dir}/final_model")
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=8) # Use logical CPU cores
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    args = parser.parse_args()
    main(args)