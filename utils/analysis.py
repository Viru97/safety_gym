"""
Analysis Utilities for Safe Stop Training

Provides tools for:
- Training progress visualization
- Performance metrics analysis
- Failure case identification
- Policy behavior analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict


class TrainingAnalyzer:
    """Analyze training logs and metrics."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict:
        """Load metrics from tensorboard logs."""
        # This is a placeholder - actual implementation would use tensorboard reader
        return {}
    
    def plot_learning_curves(self, save_path: Optional[str] = None):
        """Plot learning curves over training."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        # Placeholder data structure
        plots = [
            ('Episode Return', 'episode/return'),
            ('Success Rate', 'episode/success_rate'),
            ('Collision Rate', 'episode/collision_rate'),
            ('Avg Speed', 'episode/avg_speed'),
            ('Stillness Rate', 'episode/stillness_rate'),
            ('Episode Length', 'episode/length'),
        ]
        
        for idx, (title, key) in enumerate(plots):
            ax = axes[idx // 3, idx % 3]
            ax.set_title(title)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved learning curves to {save_path}")
        else:
            plt.show()
    
    def analyze_reward_components(self) -> Dict[str, float]:
        """Analyze which reward components are most influential."""
        components = {
            'collision': 0.0,
            'survival': 0.0,
            'movement': 0.0,
            'smoothness': 0.0,
            'conservative': 0.0,
            'threat_passed': 0.0,
            'safety': 0.0,
        }
        
        # Would analyze actual episode data here
        return components
    
    def identify_failure_modes(self) -> Dict[str, List]:
        """Identify common failure patterns."""
        failures = {
            'early_collision': [],  # Collisions in first 50 steps
            'late_collision': [],   # Collisions after 150 steps
            'unnecessary_movement': [],  # Movement when not threatened
            'insufficient_avoidance': [],  # Too little movement when threatened
        }
        
        return failures


class PolicyAnalyzer:
    """Analyze trained policy behavior."""
    
    def __init__(self, policy, env, device='cpu'):
        self.policy = policy
        self.env = env
        self.device = device
        self.policy.eval()
    
    def test_scenarios(self, num_scenarios: int = 20) -> Dict:
        """Test policy on diverse scenarios."""
        import torch
        
        results = {
            'close_slow': {'success': 0, 'collision': 0, 'avg_movement': 0.0},
            'close_fast': {'success': 0, 'collision': 0, 'avg_movement': 0.0},
            'far_slow': {'success': 0, 'collision': 0, 'avg_movement': 0.0},
            'far_fast': {'success': 0, 'collision': 0, 'avg_movement': 0.0},
        }
        
        scenarios = [
            ('close_slow', (6.0, 8.0), (-1.0, -0.5)),
            ('close_fast', (6.0, 8.0), (-2.0, -1.5)),
            ('far_slow', (12.0, 15.0), (-1.0, -0.5)),
            ('far_fast', (12.0, 15.0), (-2.0, -1.5)),
        ]
        
        for scenario_name, spawn_range, vel_range in scenarios:
            movements = []
            
            for _ in range(num_scenarios):
                # Set scenario-specific parameters
                self.env.cfg.obj_spawn_x_range = spawn_range
                self.env.cfg.obj_vx_range = vel_range
                
                obs = self.env.reset()
                done = False
                episode_movements = []
                
                while not done:
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        action = self.policy.act_inference(obs_tensor)
                    action_val = action.cpu().numpy().flatten()[0]
                    
                    obs, reward, done, info = self.env.step(action_val)
                    episode_movements.append(abs(self.env.ego_vy))
                
                if info.get('collision', False):
                    results[scenario_name]['collision'] += 1
                else:
                    results[scenario_name]['success'] += 1
                
                movements.append(np.mean(episode_movements) if episode_movements else 0.0)
            
            results[scenario_name]['avg_movement'] = np.mean(movements)
        
        return results
    
    def analyze_conservativeness(self, num_episodes: int = 50) -> Dict:
        """Analyze how conservative the policy is."""
        import torch
        
        stillness_rates = []
        movement_when_safe = []
        movement_when_threatened = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            
            still_count = 0
            total_steps = 0
            safe_movements = []
            threat_movements = []
            
            while not done:
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.policy.act_inference(obs_tensor)
                action_val = action.cpu().numpy().flatten()[0]
                
                # Track movement relative to threat state
                speed = abs(self.env.ego_vy)
                if self.env.threat_active:
                    threat_movements.append(speed)
                else:
                    safe_movements.append(speed)
                
                if speed < self.env.cfg.stillness_threshold:
                    still_count += 1
                
                obs, reward, done, info = self.env.step(action_val)
                total_steps += 1
            
            if total_steps > 0:
                stillness_rates.append(still_count / total_steps)
            if safe_movements:
                movement_when_safe.append(np.mean(safe_movements))
            if threat_movements:
                movement_when_threatened.append(np.mean(threat_movements))
        
        return {
            'stillness_rate': np.mean(stillness_rates) if stillness_rates else 0.0,
            'movement_when_safe': np.mean(movement_when_safe) if movement_when_safe else 0.0,
            'movement_when_threatened': np.mean(movement_when_threatened) if movement_when_threatened else 0.0,
            'conservativeness_ratio': (
                np.mean(movement_when_safe) / np.mean(movement_when_threatened)
                if movement_when_safe and movement_when_threatened else 0.0
            )
        }
    
    def visualize_action_distribution(self, num_steps: int = 1000, save_path: Optional[str] = None):
        """Visualize distribution of actions taken."""
        import torch
        
        actions_safe = []
        actions_threatened = []
        
        obs = self.env.reset()
        for _ in range(num_steps):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.policy.act_inference(obs_tensor)
            action_val = action.cpu().numpy().flatten()[0]
            
            if self.env.threat_active:
                actions_threatened.append(action_val)
            else:
                actions_safe.append(action_val)
            
            obs, _, done, _ = self.env.step(action_val)
            if done:
                obs = self.env.reset()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if actions_safe:
            axes[0].hist(actions_safe, bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[0].set_title('Action Distribution (Safe)')
            axes[0].set_xlabel('Action Value')
            axes[0].set_ylabel('Frequency')
            axes[0].axvline(0, color='red', linestyle='--', label='Zero Action')
            axes[0].legend()
        
        if actions_threatened:
            axes[1].hist(actions_threatened, bins=30, alpha=0.7, color='orange', edgecolor='black')
            axes[1].set_title('Action Distribution (Threatened)')
            axes[1].set_xlabel('Action Value')
            axes[1].set_ylabel('Frequency')
            axes[1].axvline(0, color='red', linestyle='--', label='Zero Action')
            axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved action distribution to {save_path}")
        else:
            plt.show()


class FailureCaseCollector:
    """Collect and analyze failure cases for debugging."""
    
    def __init__(self, env, policy, device='cpu'):
        self.env = env
        self.policy = policy
        self.device = device
        self.failures = []
    
    def collect_failures(self, num_episodes: int = 100, save_path: Optional[str] = None):
        """Collect failure cases with full state history."""
        import torch
        
        for ep in range(num_episodes):
            obs = self.env.reset()
            done = False
            
            episode_data = {
                'states': [],
                'actions': [],
                'rewards': [],
                'info': [],
            }
            
            while not done:
                # Record state
                episode_data['states'].append(obs.copy())
                
                # Get action
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.policy.act_inference(obs_tensor)
                action_val = action.cpu().numpy().flatten()[0]
                episode_data['actions'].append(action_val)
                
                # Step
                obs, reward, done, info = self.env.step(action_val)
                episode_data['rewards'].append(reward)
                episode_data['info'].append(info.copy())
            
            # Check if this was a failure
            if info.get('collision', False):
                self.failures.append({
                    'episode': ep,
                    'data': episode_data,
                    'reason': 'collision',
                    'step_count': len(episode_data['states']),
                })
        
        if save_path:
            # Save failures as JSON (convert numpy arrays to lists)
            failures_serializable = []
            for f in self.failures:
                failure_dict = {
                    'episode': f['episode'],
                    'reason': f['reason'],
                    'step_count': f['step_count'],
                    'final_state': f['data']['states'][-1].tolist() if f['data']['states'] else [],
                }
                failures_serializable.append(failure_dict)
            
            with open(save_path, 'w') as file:
                json.dump(failures_serializable, file, indent=2)
            
            print(f"Saved {len(self.failures)} failures to {save_path}")
        
        return self.failures
    
    def analyze_failure_patterns(self) -> Dict:
        """Analyze patterns in failure cases."""
        if not self.failures:
            return {}
        
        patterns = {
            'avg_failure_step': np.mean([f['step_count'] for f in self.failures]),
            'early_failures': sum(1 for f in self.failures if f['step_count'] < 50),
            'late_failures': sum(1 for f in self.failures if f['step_count'] >= 150),
        }
        
        return patterns


def generate_analysis_report(log_dir: str, model_path: str, output_path: str = 'analysis_report.txt'):
    """Generate comprehensive analysis report."""
    report = []
    report.append("="*80)
    report.append("SAFE STOP TRAINING ANALYSIS REPORT")
    report.append("="*80)
    report.append("")
    
    # Add analysis sections here
    report.append("1. TRAINING PROGRESS")
    report.append("-" * 80)
    report.append("   - Learning curves show...")
    report.append("")
    
    report.append("2. POLICY BEHAVIOR")
    report.append("-" * 80)
    report.append("   - Conservativeness analysis...")
    report.append("")
    
    report.append("3. FAILURE ANALYSIS")
    report.append("-" * 80)
    report.append("   - Common failure modes...")
    report.append("")
    
    report.append("4. RECOMMENDATIONS")
    report.append("-" * 80)
    report.append("   - Suggested improvements...")
    report.append("")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Analysis report saved to {output_path}")


if __name__ == "__main__":
    print("Analysis utilities loaded. Use from main scripts or notebooks.")
