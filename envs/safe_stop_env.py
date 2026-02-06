"""
Safe Stop Environment - Complete Implementation

This module provides a conservative lateral avoidance environment where:
- Ego robot controls only lateral velocity (vy)
- Policy learns to avoid collisions with dynamic obstacles
- Conservative behavior: "don't move unless necessary to survive"
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
    dt: float = 1/8  # 8Hz observation/control frequency
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
    threat_threshold: float = 0.5  # meters, below this d_min = threat (increased)
    stillness_threshold: float = 0.1  # m/s
    safe_distance: float = 2.0  # meters, safe clearance distance (INCREASED - key parameter!)

    # Reward weights
    collision_penalty: float = -1000.0
    survival_reward: float = 1.0
    movement_penalty_scale: float = 1.0
    action_smoothness_scale: float = 1.0
    threat_passage_reward: float = 10.0
    conservative_bonus_scale: float = 0.3


class BaseEnv(ABC):
    """
    Abstract base environment for lateral avoidance scenarios.
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

        # Dimensions
        self.obs_dim = self._get_obs_dim()
        self.action_dim = 1

        # Reward logging
        self._last_reward_components: Dict[str, float] = {}

    def _get_obs_dim(self) -> int:
        """Get observation dimension."""
        # 14-dim observation:
        # [0-1]: ego position (x, y)
        # [2]: ego lateral velocity (vy)
        # [3-4]: relative object position (dx, dy)
        # [5-6]: relative object velocity (dvx, dvy)
        # [7]: distance to object
        # [8]: time to closest approach (TTCA)
        # [9]: minimum distance at CPA (d_min)
        # [10]: threat_active flag
        # [11]: object in FOV flag
        # [12]: normalized step count
        # [13]: previous action
        return 14

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        # Reset ego
        self.ego_pos = np.array(self.cfg.ego_start_pos, dtype=np.float32)
        self.ego_vy = 0.0
        self.ego_vy_target = 0.0
        self.prev_action = 0.0

        self.collision_occurred = False  # ADD THIS
        # Spawn object
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
        """Execute one environment step."""
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

        # Compute reward
        reward = self._compute_reward(action, collision)

        # Check termination
        done, term_reason = self._check_termination(collision)
        self.termination_reason = term_reason

        # Update step count
        self.step_count += 1
        self.prev_action = action

        # Build info dict
        info = self._build_info(action, collision)

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

    def _update_threat_state(self):
        """Update threat tracking state."""
        if not self._is_object_in_fov():
            return

        _, d_min = self._compute_ttca_and_dmin()

        # Check if object passed by checking x-position
        if self.obj_pos[0] < self.ego_pos[0] - 0.5:
            if self.threat_active and not self.threat_passed:
                self.threat_passed = True
            self.threat_active = False
            return

        # Activate threat if d_min is below threshold
        if d_min < self.cfg.threat_threshold:
            self.threat_active = True
        else:
            self.threat_active = False

    def _build_info(self, action: float, collision: bool) -> Dict[str, Any]:
        """Build info dictionary."""
        ttca, d_min = self._compute_ttca_and_dmin()
        return {
            "collision": collision,
            "threat_active": self.threat_active,
            "threat_passed": self.threat_passed,
            "cooldown_counter": self.cooldown_counter,
            "termination_reason": self.termination_reason.value,
            "ttca": ttca,
            "d_min": d_min,
            "ego_speed": abs(self.ego_vy),
            "reward_components": self._last_reward_components.copy(),
        }

    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def _spawn_object(self):
        """Spawn the dynamic object. Must be implemented by subclass."""
        pass

    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Get current observation. Must be implemented by subclass."""
        pass

    @abstractmethod
    def _compute_reward(self, action: float, collision: bool) -> float:
        """Compute step reward. Must be implemented by subclass."""
        pass

    @abstractmethod
    def _check_termination(self, collision: bool) -> Tuple[bool, TerminationReason]:
        """Check if episode should terminate. Must be implemented by subclass."""
        pass

    @abstractmethod
    def _on_reset(self):
        """Additional reset logic. Must be implemented by subclass."""
        pass


class ConservativeAvoidanceEnv(BaseEnv):
    """
    Conservative lateral avoidance environment.

    The policy learns to:
    1. Stay still when no threat is present
    2. Make minimal movements to avoid collisions
    3. Return to stillness after threat passes
    """

    def _spawn_object(self):
        """Spawn object with random position and velocity."""
        x = np.random.uniform(*self.cfg.obj_spawn_x_range)
        y = np.random.uniform(*self.cfg.obj_spawn_y_range)
        self.obj_pos = np.array([x, y], dtype=np.float32)

        vx = np.random.uniform(*self.cfg.obj_vx_range)
        vy = np.random.uniform(*self.cfg.obj_vy_range)
        self.obj_vel = np.array([vx, vy], dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        """
        Get 14-dimensional observation vector.
        """
        rel_pos = self.obj_pos - self.ego_pos
        rel_vel = self.obj_vel - np.array([0.0, self.ego_vy])
        distance = np.linalg.norm(rel_pos)
        ttca, d_min = self._compute_ttca_and_dmin()

        obs = np.array([
            self.ego_pos[0],  # 0: ego x
            self.ego_pos[1],  # 1: ego y
            self.ego_vy,  # 2: ego lateral velocity
            rel_pos[0],  # 3: relative object x
            rel_pos[1],  # 4: relative object y
            rel_vel[0],  # 5: relative object vx
            rel_vel[1],  # 6: relative object vy
            distance,  # 7: distance to object
            ttca,  # 8: time to closest approach
            d_min,  # 9: minimum distance at CPA
            float(self.threat_active),  # 10: threat flag
            float(self._is_object_in_fov()),  # 11: in FOV flag
            self.step_count / self.cfg.max_episode_steps,  # 12: normalized time
            self.prev_action,  # 13: previous action
        ], dtype=np.float32)

        # Clip extreme values for numerical stability
        obs = np.clip(obs, -100.0, 100.0)

        return obs

    def _compute_reward(self, action: float, collision: bool) -> float:
        """
        FIXED Conservative reward design:

        KEY PRINCIPLE: Only penalize movement when there's NO real threat!
        If d_min is large (object will pass safely), heavily reward stillness.
        """
        reward = 0.0
        components = {}

        # 1. Collision penalty (terminal)
        if collision:
            components['collision'] = self.cfg.collision_penalty
            reward += components['collision']
            self._last_reward_components = components
            return reward

        # Get threat information
        ttca, d_min = self._compute_ttca_and_dmin()
        speed = abs(self.ego_vy)
        is_still = speed < self.cfg.stillness_threshold

        # 2. Survival reward (base reward for each step alive)
        components['survival'] = self.cfg.survival_reward
        reward += components['survival']

        # 3. CRITICAL: Distinguish between safe and dangerous scenarios
        # If d_min > safe_distance, object will pass safely - STAY STILL!
        if d_min > self.cfg.safe_distance and self._is_object_in_fov():
            # SAFE SCENARIO - heavily reward stillness, heavily penalize movement
            if is_still:
                # Big reward for staying still when safe
                still_bonus = 5.0
                components['safe_still'] = still_bonus
                reward += still_bonus
            else:
                # Big penalty for moving when safe
                unnecessary_movement_penalty = -10.0 * speed
                components['safe_moving'] = unnecessary_movement_penalty
                reward += unnecessary_movement_penalty

            components['movement'] = 0.0  # Don't double-penalize

        elif d_min < self.cfg.safe_distance and self._is_object_in_fov():
            # DANGER SCENARIO - allow movement, but prefer efficiency
            # Small penalty for movement (encourage minimal avoidance)
            movement_penalty = -0.5 * speed
            components['movement'] = movement_penalty
            reward += movement_penalty

            components['safe_still'] = 0.0
            components['safe_moving'] = 0.0

        else:
            # NO OBJECT IN FOV - definitely stay still
            if is_still:
                still_bonus = 3.0
                components['no_obj_still'] = still_bonus
                reward += still_bonus
            else:
                unnecessary_penalty = -5.0 * speed
                components['no_obj_moving'] = unnecessary_penalty
                reward += unnecessary_penalty

            components['movement'] = 0.0
            components['safe_still'] = 0.0
            components['safe_moving'] = 0.0

        # 4. Action smoothness - penalize jerky movements
        action_change = abs(action - self.prev_action)
        smoothness_penalty = -0.5 * action_change  # Linear, not quadratic
        components['smoothness'] = smoothness_penalty
        reward += smoothness_penalty

        # 5. Threat passage reward - bonus for successfully avoiding
        if self.threat_passed and not self.collision_occurred:
            components['threat_passed'] = self.cfg.threat_passage_reward
            reward += components['threat_passed']
            self.threat_passed = False  # Reset flag
        else:
            components['threat_passed'] = 0.0

        self._last_reward_components = components
        return reward

    # def _compute_reward(self, action: float, collision: bool) -> float:
    #     """Simple reward function for testing."""
    #     if collision:
    #         return -1000.0
    #
    #     reward = 0.1
    #
    #     # Penalize movement when no threat
    #     if not self.threat_active:
    #         reward -= 1.0 * action * action
    #
    #     self._last_reward_components = {"base": reward}
    #     return reward

    def _check_termination(self, collision: bool) -> Tuple[bool, TerminationReason]:
        """Check termination conditions."""
        # Collision
        if collision:
            return True, TerminationReason.COLLISION

        # Timeout
        if self.step_count >= self.cfg.max_episode_steps:
            return True, TerminationReason.TIMEOUT

        # Out of bounds
        if (self.ego_pos[1] < self.cfg.arena_y_range[0] or
            self.ego_pos[1] > self.cfg.arena_y_range[1]):
            return True, TerminationReason.COLLISION

        # Success: object passed safely
        if self.obj_pos[0] < self.cfg.arena_x_range[0] and not self.collision_occurred:
            return True, TerminationReason.SUCCESS

        return False, TerminationReason.NONE

    def _on_reset(self):
        """Additional reset logic."""
        pass


# Alias for backward compatibility
SimpleTestEnv = ConservativeAvoidanceEnv


class EnvRenderer:
    """Pygame-based visualization for the environment."""

    def __init__(self, env: BaseEnv, width: int = 1200, height: int = 800, scale: float = 20.0):
        pygame.init()

        self.env = env
        self.width = width
        self.height = height
        self.scale = scale

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Safe Stop Environment")

        self.main_surface = pygame.Surface((width, height))
        self.clock = pygame.time.Clock()

        # Calculate offsets to center the view
        self.offset_x = width // 2
        self.offset_y = height // 2

        # Colors
        self.colors = {
            "background": (20, 20, 25),
            "grid": (40, 40, 45),
            "ego_safe": (50, 200, 50),
            "ego_moving": (200, 200, 50),
            "ego_danger": (255, 100, 50),
            "object": (220, 50, 50),
            "fov": (100, 100, 150, 50),
            "text": (200, 200, 200),
        }

        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

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
        """Draw the ego robot with collision visualization."""
        cfg = self.env.cfg
        surface = self.main_surface

        pos = self.world_to_screen(*self.env.ego_pos)
        radius = int(cfg.ego_radius * self.scale)

        # Color logic with collision detection
        if self.env.collision_occurred:
            # COLLISION STATE: Flash red/white
            color = (255, 50, 50) if int(self.env.step_count * 10) % 2 else (255, 255, 255)
        elif self.env.threat_active:
            color = self.colors["ego_danger"]  # Orange-red
        elif abs(self.env.ego_vy) > cfg.stillness_threshold:
            color = self.colors["ego_moving"]  # Yellow
        else:
            color = self.colors["ego_safe"]  # Green

        # Draw ego circle
        pygame.draw.circle(surface, color, pos, radius)
        pygame.draw.circle(surface, (255, 255, 255), pos, radius, 2)

        # Collision explosion effect
        if self.env.collision_occurred:
            explosion_radius = radius + int(10 * math.sin(self.env.step_count * 0.3))
            pygame.draw.circle(surface, (255, 100, 0, 100), pos, explosion_radius, 5)

        # Velocity indicator (only if not collided)
        if abs(self.env.ego_vy) > 0.01 and not self.env.collision_occurred:
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

    def draw_info(self, reward: float = 0.0, action: float = 0.0) -> None:
        """Draw information overlay."""
        surface = self.main_surface
        y_offset = 10
        line_height = 25

        collision_status = "COLLISION!" if self.env.collision_occurred else "Safe"
        threat_status = "ACTIVE" if self.env.threat_active else "None"

        info_texts = [
            f"Step: {self.env.step_count}/{self.env.cfg.max_episode_steps}",
            f"Reward: {reward:.2f}",
            f"Action: {action:.3f}",
            f"Ego vy: {self.env.ego_vy:.3f} m/s",
            f"Threat: {threat_status}",
            f"Collision: {collision_status}",
            f"Status: {self.env.termination_reason.value}",
        ]

        for i, text in enumerate(info_texts):
            color = (255, 100, 100) if "COLLISION!" in text else self.colors["text"]
            rendered = self.font.render(text, True, color)
            surface.blit(rendered, (10, y_offset + i * line_height))

        # Draw reward components
        if hasattr(self.env, '_last_reward_components') and self.env._last_reward_components:
            y_offset = 10
            x_offset = self.width - 280
            comp_text = self.font.render("Reward Components:", True, self.colors["text"])
            surface.blit(comp_text, (x_offset, y_offset))
            y_offset += line_height

            for key, value in self.env._last_reward_components.items():
                text = self.small_font.render(f"{key}: {value:.3f}", True, self.colors["text"])
                surface.blit(text, (x_offset, y_offset))
                y_offset += 20

    def render(self, reward: float = 0.0, action: float = 0.0) -> pygame.Surface:
        """Render the current environment state."""
        self.main_surface.fill(self.colors["background"])

        self.draw_grid()
        self.draw_fov()
        self.draw_object()
        self.draw_ego()
        self.draw_info(reward, action)

        self.screen.blit(self.main_surface, (0, 0))
        pygame.display.flip()

        return self.main_surface

    def handle_events(self) -> Tuple[Optional[float], bool, bool]:
        """Handle pygame events and keyboard input."""
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