import sys
import os
import argparse
import numpy as np

# Tell Python to look in the root directory for packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from envs.safe_stop_env import SafeStopEnv


def main(args):
    # Initialize environment in human mode so the renderer is active
    env = SafeStopEnv(render_mode="human")

    print(f"Loading model from: {args.model_path}")
    try:
        model = PPO.load(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model path is correct.")
        sys.exit(1)

    video_writer = None
    if args.save_video:
        try:
            import imageio
            os.makedirs(os.path.dirname(args.video_path), exist_ok=True)
            # Create a video writer at 30 FPS
            video_writer = imageio.get_writer(args.video_path, fps=30)
            print(f"Recording enabled. Saving video to {args.video_path}...")
        except ImportError:
            print("Error: 'imageio' is required to save videos.")
            print("Please install it running: pip install imageio[ffmpeg]")
            sys.exit(1)

    obs, info = env.reset()

    print("Starting evaluation... (Close the Pygame window or press Ctrl+C to stop)")
    try:
        for step in range(args.max_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Capture frame if saving video
            if args.save_video and env.renderer:
                import pygame
                # Pygame surfaces are (Width, Height, Channels)
                # ImageIO expects (Height, Width, Channels), so we transpose the first two axes
                frame = pygame.surfarray.array3d(env.renderer.main_surface)
                frame = np.transpose(frame, (1, 0, 2))
                video_writer.append_data(frame)

            if terminated or truncated:
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")

    # Cleanup and save
    if video_writer:
        video_writer.close()
        print(f"✅ Video successfully saved to {args.video_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Safe Stop Agent")
    parser.add_argument("--model_path", type=str, default="./logs/safe_stop_ppo/final_model.zip",
                        help="Path to the trained model (.zip file)")
    parser.add_argument("--max_steps", type=int, default=5000,
                        help="Maximum number of steps to evaluate")
    parser.add_argument("--save_video", action="store_true",
                        help="Flag to save the evaluation to a video file")
    parser.add_argument("--video_path", type=str, default="./assets/evaluation.mp4",
                        help="Path where the video will be saved")

    args = parser.parse_args()
    main(args)