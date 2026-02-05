#!/usr/bin/env python3
"""
Quick Start Guide for Safe Stop Training

This script provides an interactive guide to get started with training.
"""

import os
import sys


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def print_step(num, text):
    """Print formatted step."""
    print(f"\n[Step {num}] {text}")
    print("-" * 70)


def check_dependencies():
    """Check if required packages are installed."""
    print_step(1, "Checking Dependencies")
    
    required = {
        'numpy': 'numpy',
        'torch': 'torch',
        'pygame': 'pygame',
        'rsl_rl': 'rsl_rl',
    }
    
    missing = []
    for name, package in required.items():
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    else:
        print("\n✓ All dependencies satisfied!")
        return True


def test_environment():
    """Test basic environment functionality."""
    print_step(2, "Testing Environment")
    
    try:
        from envs.safe_stop_env import ConservativeAvoidanceEnv, BaseEnvConfig
        
        config = BaseEnvConfig(max_episode_steps=10)
        env = ConservativeAvoidanceEnv(config)
        
        obs = env.reset()
        print(f"✓ Environment created successfully")
        print(f"  Observation dim: {len(obs)}")
        
        for i in range(5):
            action = 0.0
            obs, reward, done, info = env.step(action)
            if done:
                break
        
        print(f"✓ Environment step working")
        print(f"  Sample reward: {reward:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_training_options():
    """Show training command options."""
    print_step(3, "Training Options")
    
    print("Quick Start (Small Scale):")
    print("  python train.py --mode train --n_envs 64 --total_timesteps 1000000")
    print()
    
    print("Recommended (Full Training):")
    print("  python train.py --mode train --n_envs 256 --total_timesteps 10000000 --use_curriculum")
    print()
    
    print("Fast Training (CPU):")
    print("  python train.py --mode train --n_envs 32 --total_timesteps 500000 --cpu")
    print()
    
    print("With Custom Hyperparameters:")
    print("  python train.py --mode train --learning_rate 1e-4 --ent_coef 0.02")
    print()


def show_evaluation_options():
    """Show evaluation command options."""
    print_step(4, "Evaluation Options")
    
    print("Evaluate with Rendering:")
    print("  python train.py --mode eval --model_path logs/run_*/final_model.pt --render")
    print()
    
    print("Evaluate Multiple Episodes:")
    print("  python train.py --mode eval --model_path logs/run_*/final_model.pt --num_episodes 50")
    print()
    
    print("Headless Evaluation:")
    print("  python train.py --mode eval --model_path logs/run_*/final_model.pt --num_episodes 100")
    print()


def show_analysis_options():
    """Show analysis options."""
    print_step(5, "Analysis & Debugging")
    
    print("Run Environment Tests:")
    print("  python test_environment.py")
    print()
    
    print("Analyze Training Progress (in Python):")
    print("""
  from utils.analysis import TrainingAnalyzer
  analyzer = TrainingAnalyzer('logs/run_TIMESTAMP')
  analyzer.plot_learning_curves(save_path='curves.png')
    """)
    
    print("\nMonitor with Tensorboard:")
    print("  tensorboard --logdir logs")
    print()


def show_troubleshooting():
    """Show common issues and solutions."""
    print_step(6, "Common Issues & Solutions")
    
    issues = [
        (
            "Training is unstable / policy diverges",
            [
                "Enable curriculum learning: --use_curriculum",
                "Reduce learning rate: --learning_rate 1e-4",
                "Increase num environments: --n_envs 512",
            ]
        ),
        (
            "Policy moves too much (not conservative)",
            [
                "Check reward components in environment",
                "Increase movement penalty in BaseEnvConfig",
                "Verify conservative bonus is being awarded",
            ]
        ),
        (
            "High collision rate",
            [
                "Train longer: increase --total_timesteps",
                "Enable curriculum learning",
                "Check threat detection logic",
            ]
        ),
        (
            "Out of memory errors",
            [
                "Reduce num environments: --n_envs 128",
                "Use CPU: --cpu",
                "Reduce batch size: --num_mini_batches 2",
            ]
        ),
    ]
    
    for issue, solutions in issues:
        print(f"\n{issue}:")
        for sol in solutions:
            print(f"  • {sol}")


def main():
    """Main quick start guide."""
    print_header("SAFE STOP - Quick Start Guide")
    
    print("This guide will help you get started with training a conservative")
    print("lateral avoidance policy using reinforcement learning.")
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n⚠ Please install missing dependencies before continuing.")
        sys.exit(1)
    
    # Test environment
    env_ok = test_environment()
    
    if not env_ok:
        print("\n⚠ Environment test failed. Please check the installation.")
        sys.exit(1)
    
    # Show options
    show_training_options()
    show_evaluation_options()
    show_analysis_options()
    show_troubleshooting()
    
    # Next steps
    print_header("Next Steps")
    
    print("1. Run environment tests:")
    print("   python test_environment.py")
    print()
    
    print("2. Start training (quick test):")
    print("   python train.py --mode train --n_envs 64 --total_timesteps 1000000")
    print()
    
    print("3. Monitor training:")
    print("   tensorboard --logdir logs")
    print()
    
    print("4. Evaluate trained model:")
    print("   python train.py --mode eval --model_path logs/run_*/final_model.pt --render")
    print()
    
    print("5. Analyze results:")
    print("   python -c \"from utils.analysis import TrainingAnalyzer; analyzer = TrainingAnalyzer('logs/run_TIMESTAMP'); analyzer.plot_learning_curves()\"")
    print()
    
    print_header("Resources")
    
    print("Documentation: README.md")
    print("Environment code: envs/safe_stop_env.py")
    print("Training code: train.py")
    print("Analysis tools: utils/analysis.py")
    print()
    
    print("For detailed information, see the README.md file.")
    print()


if __name__ == "__main__":
    try:
        main()
        print("✓ Quick start guide completed successfully!")
        print()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
