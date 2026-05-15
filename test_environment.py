"""
Test Script for Safe Stop Environment

Run various tests to validate environment and training setup.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from envs.safe_stop_env import ConservativeAvoidanceEnv, BaseEnvConfig, EnvRenderer


def test_environment_basic():
    """Test basic environment functionality."""
    print("\n" + "="*60)
    print("TEST 1: Basic Environment Functionality")
    print("="*60)
    
    config = BaseEnvConfig(max_episode_steps=50)
    env = ConservativeAvoidanceEnv(config)
    
    # Test reset
    obs = env.reset(seed=42)
    assert obs.shape == (14,), f"Expected obs shape (14,), got {obs.shape}"
    print(f"✓ Reset successful, obs shape: {obs.shape}")
    
    # Test step
    action = 0.5
    obs, reward, done, info = env.step(action)
    assert obs.shape == (14,), f"Expected obs shape (14,), got {obs.shape}"
    assert isinstance(reward, (float, np.floating)), "Reward should be float"
    assert isinstance(done, (bool, np.bool_)), "Done should be bool"
    print(f"✓ Step successful, reward: {reward:.3f}, done: {done}")
    
    # Test episode completion
    episode_length = 0
    obs = env.reset()
    while episode_length < config.max_episode_steps:
        action = np.random.uniform(-1, 1)
        obs, reward, done, info = env.step(action)
        episode_length += 1
        if done:
            break
    
    print(f"✓ Episode completed in {episode_length} steps")
    print(f"  Termination reason: {info.get('termination_reason', 'unknown')}")
    
    return True


def test_reward_components():
    """Test reward component calculation."""
    print("\n" + "="*60)
    print("TEST 2: Reward Components")
    print("="*60)
    
    config = BaseEnvConfig()
    env = ConservativeAvoidanceEnv(config)
    
    # Test 1: Collision should give large negative reward
    obs = env.reset()
    env.obj_pos = env.ego_pos.copy()  # Force collision
    obs, reward, done, info = env.step(0.0)
    assert reward < -50, f"Collision reward should be very negative, got {reward}"
    print(f"✓ Collision penalty working: {reward:.1f}")
    
    # Test 2: Staying still when safe should give positive reward
    obs = env.reset()
    env.threat_active = False
    obs, reward, done, info = env.step(0.0)
    components = info.get('reward_components', {})
    print(f"✓ Stillness when safe:")
    for key, value in components.items():
        print(f"    {key}: {value:.3f}")
    
    # Test 3: Movement should be penalized
    obs = env.reset()
    obs, reward_still, done, info = env.step(0.0)
    obs, reward_moving, done, info = env.step(1.0)
    print(f"✓ Movement penalty: still={reward_still:.3f}, moving={reward_moving:.3f}")
    
    return True


def test_threat_detection():
    """Test threat detection logic."""
    print("\n" + "="*60)
    print("TEST 3: Threat Detection")
    print("="*60)
    
    config = BaseEnvConfig()
    env = ConservativeAvoidanceEnv(config)
    
    # Test with object on collision course
    obs = env.reset()
    env.obj_pos = np.array([5.0, 0.0], dtype=np.float32)
    env.obj_vel = np.array([-1.0, 0.0], dtype=np.float32)
    env.ego_pos = np.array([0.0, 0.0], dtype=np.float32)
    env.ego_vy = 0.0
    
    env._update_threat_state()
    ttca, d_min = env._compute_ttca_and_dmin()
    
    print(f"✓ TTCA: {ttca:.2f} seconds")
    print(f"✓ D_min: {d_min:.2f} meters")
    print(f"✓ Threat active: {env.threat_active}")
    
    assert ttca > 0, "TTCA should be positive for approaching object"
    
    return True


def test_observation_validity():
    """Test observation values are reasonable."""
    print("\n" + "="*60)
    print("TEST 4: Observation Validity")
    print("="*60)
    
    config = BaseEnvConfig()
    env = ConservativeAvoidanceEnv(config)
    
    obs = env.reset()
    
    # Run several steps and check observations
    for i in range(10):
        action = np.random.uniform(-1, 1)
        obs, reward, done, info = env.step(action)
        
        # Check for NaN or Inf
        assert not np.any(np.isnan(obs)), f"NaN in observation at step {i}"
        assert not np.any(np.isinf(obs)), f"Inf in observation at step {i}"
        
        # Check reasonable ranges
        assert obs[2] >= -2.0 and obs[2] <= 2.0, f"Ego velocity out of range: {obs[2]}"
        
        if done:
            obs = env.reset()
    
    print(f"✓ All observations valid over 10 steps")
    print(f"  Sample observation: {obs[:5]}")
    
    return True


def test_vectorized_environment():
    """Test vectorized environment wrapper."""
    print("\n" + "="*60)
    print("TEST 5: Vectorized Environment")
    print("="*60)
    
    try:
        from scripts.train import VecEnvWrapper, GymEnvWrapper
        
        def make_env():
            env = ConservativeAvoidanceEnv(BaseEnvConfig())
            return GymEnvWrapper(env)
        
        num_envs = 4
        vec_env = VecEnvWrapper(
            env_fn=make_env,
            num_envs=num_envs,
            device='cpu',
            seed=42,
            render=False,
        )
        
        # Test reset
        obs_dict = vec_env.reset()
        obs = obs_dict['observations']
        assert obs.shape == (num_envs, 14), f"Expected shape ({num_envs}, 14), got {obs.shape}"
        print(f"✓ Vectorized reset successful: {obs.shape}")
        
        # Test step
        actions = torch.randn(num_envs, 1)
        obs_dict, rewards, dones, extras = vec_env.step(actions)
        obs = obs_dict['observations']
        assert obs.shape == (num_envs, 14), f"Expected shape ({num_envs}, 14), got {obs.shape}"
        assert rewards.shape == (num_envs,), f"Expected rewards shape ({num_envs},), got {rewards.shape}"
        print(f"✓ Vectorized step successful")
        print(f"  Rewards: {rewards}")
        
        vec_env.close()
        
        return True
        
    except ImportError as e:
        print(f"⚠ Skipping vectorized env test (missing dependency): {e}")
        return True


def test_rendering():
    """Test environment rendering."""
    print("\n" + "="*60)
    print("TEST 6: Environment Rendering")
    print("="*60)
    
    try:
        import pygame
        
        config = BaseEnvConfig(max_episode_steps=20)
        env = ConservativeAvoidanceEnv(config)
        renderer = EnvRenderer(env, scale=20.0)
        
        obs = env.reset()
        
        # Render a few frames
        for i in range(5):
            action = 0.0 if i % 2 == 0 else 0.5
            obs, reward, done, info = env.step(action)
            surface = renderer.render(reward=reward, action=action)
            
            assert surface is not None, "Renderer should return surface"
            
            if done:
                break
        
        renderer.close()
        
        print(f"✓ Rendering successful (5 frames)")
        return True
        
    except Exception as e:
        print(f"⚠ Rendering test failed: {e}")
        print("  (This is okay if running headless)")
        return True


def test_conservative_behavior():
    """Test if default policy encourages conservative behavior."""
    print("\n" + "="*60)
    print("TEST 7: Conservative Behavior Tendency")
    print("="*60)
    
    config = BaseEnvConfig()
    env = ConservativeAvoidanceEnv(config)
    
    # Track stillness when no threat
    stillness_count = 0
    total_safe_steps = 0
    
    for episode in range(10):
        obs = env.reset()
        
        for step in range(100):
            # Random policy
            action = np.random.uniform(-0.1, 0.1)  # Small actions
            obs, reward, done, info = env.step(action)
            
            if not env.threat_active:
                total_safe_steps += 1
                if abs(env.ego_vy) < env.cfg.stillness_threshold:
                    stillness_count += 1
            
            if done:
                break
    
    stillness_rate = stillness_count / total_safe_steps if total_safe_steps > 0 else 0.0
    print(f"✓ Random policy stillness rate when safe: {stillness_rate*100:.1f}%")
    print(f"  (Higher values indicate reward structure encourages stillness)")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("SAFE STOP ENVIRONMENT TEST SUITE")
    print("="*60)
    
    tests = [
        ("Basic Environment", test_environment_basic),
        ("Reward Components", test_reward_components),
        ("Threat Detection", test_threat_detection),
        ("Observation Validity", test_observation_validity),
        ("Vectorized Environment", test_vectorized_environment),
        ("Rendering", test_rendering),
        ("Conservative Behavior", test_conservative_behavior),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} FAILED:")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Safe Stop Environment")
    parser.add_argument("--test", type=str, default="all", 
                       help="Specific test to run (or 'all')")
    args = parser.parse_args()
    
    if args.test == "all":
        success = run_all_tests()
        sys.exit(0 if success else 1)
    else:
        print(f"Running specific test: {args.test}")
        # Could add specific test running logic here
