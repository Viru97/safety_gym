"""
Safe Stop Environment V1 - Base Environment and Renderer Test

This module provides:
1. BaseEnvConfig - Configuration dataclass with common parameters
2. BaseEnv - Abstract base environment with placeholder methods for inheritance
3. SimpleTestEnv - Concrete implementation for testing EnvRenderer
4. EnvRenderer - Pygame-based visualization
5. Main interface to test the renderer works correctly
"""

import numpy as np
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List, Any
from enum import Enum
import pygame



class TerminationReason(Enum):
    """Reasons for episode termination."""
    NONE = "none"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    SUCCESS = "success"

    
@dataclass
class BaseEnvConfig:
    """Base environment configuration parameters."""
    
    # Timing
    dt: float = 0.125  # 8Hz observation/control frequency
    max_episode_steps: int = 200
    
    # Arena bounds
    arena_x_range: Tuple[float, float] = (-5.0, 30.0)
    arena_y_range: Tuple[float, float] = (-15.0, 15.0)
    
    # Ego robot
    ego_radius: float = 0.25
    ego_vy_max: float = 1.0  # m/s, action range [-1, 1]
    ego_max_acceleration: float = 2.0  # m/s², max acceleration/deceleration
    ego_start_pos: Tuple[float, float] = (0.0, 0.0)
    
    # Object
    obj_radius: float = 0.3
    obj_spawn_x_range: Tuple[float, float] = (6.0, 15.0)
    obj_spawn_y_range: Tuple[float, float] = (-8.0, 8.0)
    obj_vx_range: Tuple[float, float] = (-2.0, -0.5)
    obj_vy_range: Tuple[float, float] = (-0.5, 0.5)
    
    # FOV
    fov_angle: float = 130.0  # degrees
    fov_range: float = 20.0   # meters
    
    # Thresholds
    threat_threshold: float = 0.2
    stillness_threshold: float = 0.1  # m/s



class BaseEnv(ABC):
    """
    Abstract base environment for lateral avoidance scenarios.
    
    Subclasses must implement the abstract methods to define specific behaviors.
    This base class handles common functionality like state management and physics.
    """
    
    def __init__(self, config: Optional[BaseEnvConfig] = None):
        self.cfg = config or BaseEnvConfig()
        
        # State variables
        self.ego_pos = np.zeros(2, dtype=np.float32)
        self.ego_vy = 0.0
        self.ego_vy_target = 0.0
        
        self.obj_pos = np.zeros(2, dtype=np.float32)
        self.obj_vel = np.zeros(2, dtype=np.float32)
        
        self.step_count = 0
        self.prev_action = 0.0
        
        self.threat_active = False
        self.threat_passed = False
        self.cooldown_counter = 0
        self.collision_occurred = False
        self.termination_reason = TerminationReason.NONE
        
        # Dimensions (can be overridden by subclasses)
        self.obs_dim = self._get_obs_dim()
        self.action_dim = 1
        
        # Reward logging
        self._last_reward_components: Dict[str, float] = {}
    
    def _get_obs_dim(self) -> int:
        """Get observation dimension. Override in subclass if needed."""
        obs_dim = 14 # placeholder
        return obs_dim
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset ego
        self.ego_pos = np.array(self.cfg.ego_start_pos, dtype=np.float32)
        self.ego_vy = 0.0
        self.ego_vy_target = 0.0
        self.prev_action = 0.0
        
        # Spawn object (subclass implements specific behavior)
        self._spawn_object()
        
        # Reset tracking
        self.step_count = 0
        self.threat_passed = False
        self.cooldown_counter = 0
        self.collision_occurred = False
        self.termination_reason = TerminationReason.NONE
        
        # Reset threat state
        self.threat_active = False
        self._update_threat_state()
        
        # Subclass-specific reset
        self._on_reset()
        
        return self._get_observation()
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Lateral velocity command vy ∈ [-1, 1]
            
        Returns:
            observation, reward, done, info
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        self.ego_vy_target = action * self.cfg.ego_vy_max
        
        # Velocity dynamics with acceleration limits
        max_delta_v = self.cfg.ego_max_acceleration * self.cfg.dt
        velocity_error = self.ego_vy_target - self.ego_vy
        velocity_change = np.clip(velocity_error, -max_delta_v, max_delta_v)
        self.ego_vy += velocity_change
        
        # Update ego position (only y changes)
        self.ego_pos[1] += self.ego_vy * self.cfg.dt
        
        # Update object position
        self.obj_pos += self.obj_vel * self.cfg.dt
        
        # Check collision
        collision = self._check_collision()
        
        # Update threat tracking
        self._update_threat_state()
        
        # Compute reward (subclass implements)
        reward = self._compute_reward(action, collision)
        
        # Check termination (subclass implements)
        done, term_reason = self._check_termination(collision)
        self.termination_reason = term_reason
        
        # Update step count
        self.step_count += 1
        self.prev_action = action
        
        # Build info dict
        info = {}
        
        obs = self._get_observation()
        
        return obs, reward, done, info
    
    def _check_collision(self) -> bool:
        """Check if ego and object are colliding."""
        distance = np.linalg.norm(self.obj_pos - self.ego_pos)
        collision_dist = self.cfg.ego_radius + self.cfg.obj_radius
        
        if distance < collision_dist:
            self.collision_occurred = True
            return True
        return False
    
    def _is_object_in_fov(self) -> bool:
        """Check if object is within ego's field of view."""
        rel_pos = self.obj_pos - self.ego_pos
        distance = np.linalg.norm(rel_pos)
        
        if distance > self.cfg.fov_range:
            return False
        
        if distance < 1e-6:
            return True
        
        # Angle from forward direction (+x)
        angle = math.atan2(abs(rel_pos[1]), rel_pos[0])
        half_fov_rad = math.radians(self.cfg.fov_angle / 2)
        
        return rel_pos[0] > -self.cfg.ego_radius and angle <= half_fov_rad
    
    def _compute_ttca_and_dmin(self) -> Tuple[float, float]:
        """Compute Time-To-Closest-Approach and minimum distance."""
        rel_pos = self.obj_pos - self.ego_pos
        rel_vel = self.obj_vel - np.array([0.0, self.ego_vy])
        
        vel_sq = np.dot(rel_vel, rel_vel)
        
        if vel_sq < 1e-6:
            return float('inf'), np.linalg.norm(rel_pos)
        
        ttca = -np.dot(rel_pos, rel_vel) / vel_sq
        ttca = max(ttca, 0)
        
        closest_pos = rel_pos + rel_vel * ttca
        d_min = np.linalg.norm(closest_pos)
        
        return ttca, d_min
    
    def _build_info(self, action: float, collision: bool) -> Dict[str, Any]:
        """Build info dictionary. Override for additional info."""
        ttca, d_min = self._compute_ttca_and_dmin()
        return {
            "collision": collision,
            "threat_active": self.threat_active,
            "threat_passed": self.threat_passed,
            "cooldown_counter": self.cooldown_counter,
            "termination_reason": self.termination_reason.value,
            "step": self.step_count,
            "ego_vy": self.ego_vy,
            "in_fov": self._is_object_in_fov(),
            "ttca": ttca,
            "d_min": d_min,
            "reward_components": self._last_reward_components.copy(),
        }
    
    @abstractmethod
    def _spawn_object(self) -> None:
        """Spawn object in the environment. Subclass must implement."""
        pass
    
    @abstractmethod
    def _update_threat_state(self) -> None:
        """Update threat tracking state. Subclass must implement."""
        pass
    
    @abstractmethod
    def _compute_reward(self, action: float, collision: bool) -> float:
        """Compute reward for current step. Subclass must implement."""
        pass
    
    @abstractmethod
    def _check_termination(self, collision: bool) -> Tuple[bool, TerminationReason]:
        """Check if episode should terminate. Subclass must implement."""
        pass
    
    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Build observation vector. Subclass must implement."""
        pass
    
    def _on_reset(self) -> None:
        """Called at end of reset(). Override for subclass-specific reset logic."""
        pass
    
    def _compute_risk_field(self) -> float:
        """Compute risk field value. Override for custom risk calculation."""
        return 0.0
    
    def get_observation_labels(self) -> List[str]:
        """Get labels for each observation dimension. Override in subclass."""
        return [f"obs_{i}" for i in range(self.obs_dim)]


class SimpleTestEnv(BaseEnv):
    """
    Simple test environment for verifying EnvRenderer functionality.
    
    This is a minimal implementation of the base environment with basic behaviors.
    Used primarily for testing the visualization system.
    """
    
    def __init__(self, config: Optional[BaseEnvConfig] = None):
        super().__init__(config)
    
    def _spawn_object(self) -> None:
        """Spawn object with random position and velocity toward ego."""
        obj_x = np.random.uniform(*self.cfg.obj_spawn_x_range)
        obj_y = np.random.uniform(*self.cfg.obj_spawn_y_range)
        self.obj_pos = np.array([obj_x, obj_y], dtype=np.float32)
        
        # Velocity toward ego (negative x direction)
        obj_vx = np.random.uniform(*self.cfg.obj_vx_range)
        obj_vy = np.random.uniform(*self.cfg.obj_vy_range)
        self.obj_vel = np.array([obj_vx, obj_vy], dtype=np.float32)
    
    def _update_threat_state(self) -> None:
        """Simple threat state update based on FOV and distance."""
        in_fov = self._is_object_in_fov()
        ttca, d_min = self._compute_ttca_and_dmin()
        collision_radius = self.cfg.ego_radius + self.cfg.obj_radius
        
        # Threat is active if object is in FOV and will pass close
        will_collide = d_min < collision_radius + 0.5
        collision_imminent = 0 < ttca < 5.0
        
        was_threat_active = self.threat_active
        self.threat_active = in_fov and will_collide and collision_imminent
        
        # Check if threat has passed
        object_passed_ego = self.obj_pos[0] < self.ego_pos[0] - 1.0
        
        if was_threat_active and not self.threat_active:
            if object_passed_ego or not in_fov:
                self.threat_passed = True
                self.cooldown_counter = 0
        
        if self.threat_passed and not self.threat_active and object_passed_ego:
            self.cooldown_counter += 1
    
    def _compute_reward(self, action: float, collision: bool) -> float:
        """Simple reward function for testing."""
        if collision:
            return -10.0
        
        reward = 0.1 
        
        # Penalize movement when no threat
        if not self.threat_active:
            reward -= 0.5 * action * action
        
        self._last_reward_components = {"base": reward}
        return reward
    
    def _check_termination(self, collision: bool) -> Tuple[bool, TerminationReason]:
        """Check termination conditions."""
        if collision:
            return True, TerminationReason.COLLISION
        
        if self.step_count >= self.cfg.max_episode_steps:
            return True, TerminationReason.TIMEOUT
        
        return False, TerminationReason.NONE
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        in_fov = self._is_object_in_fov()
        risk = self._compute_risk_field()
        
        if in_fov:
            rel_pos = self.obj_pos - self.ego_pos
            rel_vel = self.obj_vel - np.array([0.0, self.ego_vy])
            distance = np.linalg.norm(rel_pos)
            ttca, d_min = self._compute_ttca_and_dmin()
            
            obs = np.array([
                # Ego state (3)
                self.ego_pos[0],
                self.ego_pos[1],
                self.ego_vy,
                # Object relative state (4)
                rel_pos[0],
                rel_pos[1],
                rel_vel[0],
                rel_vel[1],
                # Distance and size (2)
                distance,
                self.cfg.obj_radius,
                # Threat signals (3)
                risk,
                float(self.threat_active),
                float(in_fov),
                # Temporal signals (2)
                np.clip(ttca, 0, 10) / 10.0,
                float(self.threat_passed),
            ], dtype=np.float32)
        else:
            # Safe defaults when object not visible
            obs = np.array([
                self.ego_pos[0],
                self.ego_pos[1],
                self.ego_vy,
                self.cfg.fov_range,
                0.0,
                0.0,
                0.0,
                self.cfg.fov_range,
                self.cfg.obj_radius,
                0.0,
                0.0,
                0.0,
                1.0,
                float(self.threat_passed),
            ], dtype=np.float32)
        
        return obs
    
    def _compute_risk_field(self) -> float:
        """Compute simple risk based on distance and approach."""
        # placeholder risk calculation
        return 0.0
    
    def get_observation_labels(self) -> List[str]:
        """Get labels for each observation dimension."""
        return [
            "ego_x", "ego_y", "ego_vy",
            "obj_rel_x", "obj_rel_y", "obj_rel_vx", "obj_rel_vy",
            "obj_distance", "obj_radius",
            "risk", "threat_active", "in_fov",
            "ttca_norm", "threat_passed"
        ]



class EnvRenderer:
    """
    Pygame-based visualization for environment debugging and testing.
    
    Provides interactive visualization with keyboard controls for testing
    environment behavior and reward shaping.
    """
    
    def __init__(self, env: BaseEnv, scale: float = 20.0, 
                 replay_mode: bool = False, panel_width: int = 300):
        """
        Initialize the renderer.
        
        Args:
            env: Environment instance to visualize
            scale: Pixels per meter
            replay_mode: Whether to show replay controls panel
            panel_width: Width of side panel in replay mode
        """
        self.env = env
        self.scale = scale
        self.replay_mode = replay_mode
        self.panel_width = panel_width if replay_mode else 0
        
        cfg = env.cfg
        self.width = int((cfg.arena_x_range[1] - cfg.arena_x_range[0]) * scale)
        self.height = int((cfg.arena_y_range[1] - cfg.arena_y_range[0]) * scale)
        
        self.offset_x = -cfg.arena_x_range[0] * scale
        self.offset_y = self.height / 2
        
        # Color scheme
        self.colors = {
            "background": (30, 30, 40),
            "grid": (50, 50, 60),
            "ego_safe": (100, 200, 100),
            "ego_danger": (200, 100, 100),
            "ego_moving": (200, 200, 100),
            "object": (200, 80, 80),
            "fov": (80, 80, 120),
            "trajectory": (150, 150, 150),
            "text": (220, 220, 220),
            "threat_zone": (150, 50, 50, 100),
        }
        
        # Initialize pygame
        pygame.init()
        total_width = self.width + self.panel_width
        self.screen = pygame.display.set_mode((total_width, self.height))
        
        self.main_surface = pygame.Surface((self.width, self.height))
        if replay_mode:
            self.panel_surface = pygame.Surface((self.panel_width, self.height))
        
        title = "Environment Test - REPLAY" if replay_mode else "Environment Test"
        pygame.display.set_caption(title)
        
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        self.clock = pygame.time.Clock()
        
        # UI state
        self.slider_rect = None
        self.slider_dragging = False
    
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int(x * self.scale + self.offset_x)
        screen_y = int(self.offset_y - y * self.scale)
        return screen_x, screen_y
    
    def draw_fov(self) -> None:
        """Draw the ego's field of view cone."""
        cfg = self.env.cfg
        surface = self.main_surface
        
        ego_screen = self.world_to_screen(*self.env.ego_pos)
        half_fov_rad = math.radians(cfg.fov_angle / 2)
        
        points = [ego_screen]
        num_segments = 20
        for i in range(num_segments + 1):
            angle = -half_fov_rad + (2 * half_fov_rad) * i / num_segments
            end_x = self.env.ego_pos[0] + cfg.fov_range * math.cos(angle)
            end_y = self.env.ego_pos[1] + cfg.fov_range * math.sin(angle)
            points.append(self.world_to_screen(end_x, end_y))
        
        if len(points) >= 3:
            pygame.draw.polygon(surface, self.colors["fov"], points)
    
    def draw_grid(self) -> None:
        """Draw background grid."""
        cfg = self.env.cfg
        surface = self.main_surface
        
        # Vertical lines
        for x in range(int(cfg.arena_x_range[0]), int(cfg.arena_x_range[1]) + 1, 5):
            start = self.world_to_screen(x, cfg.arena_y_range[0])
            end = self.world_to_screen(x, cfg.arena_y_range[1])
            pygame.draw.line(surface, self.colors["grid"], start, end, 1)
        
        # Horizontal lines
        for y in range(int(cfg.arena_y_range[0]), int(cfg.arena_y_range[1]) + 1, 5):
            start = self.world_to_screen(cfg.arena_x_range[0], y)
            end = self.world_to_screen(cfg.arena_x_range[1], y)
            pygame.draw.line(surface, self.colors["grid"], start, end, 1)
        
        # Origin marker
        origin = self.world_to_screen(0, 0)
        pygame.draw.line(surface, (100, 100, 100), 
                        (origin[0] - 10, origin[1]), (origin[0] + 10, origin[1]), 2)
        pygame.draw.line(surface, (100, 100, 100),
                        (origin[0], origin[1] - 10), (origin[0], origin[1] + 10), 2)
    
    def draw_ego(self) -> None:
        """Draw the ego robot."""
        cfg = self.env.cfg
        surface = self.main_surface
        
        pos = self.world_to_screen(*self.env.ego_pos)
        radius = int(cfg.ego_radius * self.scale)
        
        # Color based on state
        if self.env.threat_active:
            color = self.colors["ego_danger"]
        elif abs(self.env.ego_vy) > cfg.stillness_threshold:
            color = self.colors["ego_moving"]
        else:
            color = self.colors["ego_safe"]
        
        pygame.draw.circle(surface, color, pos, radius)
        pygame.draw.circle(surface, (255, 255, 255), pos, radius, 2)
        
        # Velocity indicator
        if abs(self.env.ego_vy) > 0.01:
            vel_scale = 30
            end_y = pos[1] - int(self.env.ego_vy * vel_scale)
            pygame.draw.line(surface, (255, 255, 0), pos, (pos[0], end_y), 3)
            sign_vy = int(np.sign(self.env.ego_vy))
            pygame.draw.polygon(surface, (255, 255, 0), [
                (pos[0], end_y),
                (pos[0] - 5, end_y + sign_vy * 8),
                (pos[0] + 5, end_y + sign_vy * 8),
            ])

    def draw_object(self) -> None:
        """Draw the moving object."""
        cfg = self.env.cfg
        surface = self.main_surface
        
        pos = self.world_to_screen(*self.env.obj_pos)
        radius = int(cfg.obj_radius * self.scale)
        
        in_fov = self.env._is_object_in_fov()
        color = self.colors["object"] if in_fov else (100, 50, 50)
        
        pygame.draw.circle(surface, color, pos, radius)
        if in_fov:
            pygame.draw.circle(surface, (255, 100, 100), pos, radius, 2)
        
        # Velocity indicator
        vel_scale = 10
        end_x = pos[0] + int(self.env.obj_vel[0] * vel_scale)
        end_y = pos[1] - int(self.env.obj_vel[1] * vel_scale)
        pygame.draw.line(surface, (255, 150, 150), pos, (end_x, end_y), 2)
    
    def render(self, reward: float = 0.0, action: float = 0.0, 
               recording: bool = False) -> pygame.Surface:
        """
        Render the current environment state.
        
        Args:
            reward: Current step reward for display
            action: Current action for display
            recording: Whether recording indicator should be shown
            
        Returns:
            The rendered surface
        """
        self.main_surface.fill(self.colors["background"])
        
        self.draw_grid()
        self.draw_fov()
        self.draw_object()
        self.draw_ego()
        
        self.screen.blit(self.main_surface, (0, 0))
        
        pygame.display.flip()
        
        return self.main_surface
    
    def handle_events(self) -> Tuple[Optional[float], bool, bool]:
        """
        Handle pygame events and keyboard input.
        
        Returns:
            (action, reset, quit): action to take, whether to reset, whether to quit
        """
        action = None
        reset = False
        quit_game = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_game = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    quit_game = True
                elif event.key == pygame.K_SPACE:
                    reset = True
                elif event.key == pygame.K_r:
                    # Random action
                    return np.random.uniform(-1, 1), False, False
        
        # Continuous key press handling
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action = 0.5
        elif keys[pygame.K_DOWN]:
            action = -0.5
        else:
            action = 0.0
        
        return action, reset, quit_game
    
    def tick(self, fps: int = 30) -> None:
        """Limit frame rate."""
        self.clock.tick(fps)
    
    def close(self) -> None:
        """Clean up pygame resources."""
        pygame.quit()


def run_interactive_test(env_class=SimpleTestEnv, config=None):
    
    # Create environment
    env = env_class(config)
    obs = env.reset()
    
    # Create renderer
    renderer = EnvRenderer(env, scale=20.0)
    
    # Main loop
    running = True
    total_reward = 0.0
    episode_count = 0
    
    try:
        while running:
            # Handle input
            action, reset, quit_game = renderer.handle_events()
            
            if quit_game:
                running = False
                continue
            
            if reset:
                obs = env.reset()
                print(f"\n--- Episode {episode_count} ended, Total Reward: {total_reward:.2f} ---")
                total_reward = 0.0
                episode_count += 1
                continue
            
            # Step environment
            if action is not None:
                obs, reward, done, info = env.step(action)
                total_reward += reward
            else:
                action = 0.0
                reward = 0.0
            
            # Render
            renderer.render(reward=reward, action=action)
            
            # Handle episode end
            if env.termination_reason != TerminationReason.NONE:
                print(f"\n--- Episode ended: {env.termination_reason.value} ---")
                print(f"Total Reward: {total_reward:.2f}")
                obs = env.reset()
                total_reward = 0.0
                episode_count += 1
            
            # Limit frame rate
            renderer.tick(fps=30)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        renderer.close()
        print(f"\nTest completed. Total episodes: {episode_count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Environment and Renderer")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes for automated test")
    
    args = parser.parse_args()
    run_interactive_test()
