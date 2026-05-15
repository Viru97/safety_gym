"""
Safe Stop Environment - Refactored for Gymnasium & Stable Baselines3

This module provides a conservative lateral avoidance environment where:
- Ego robot controls only lateral velocity (vy)
- Policy learns to avoid collisions with dynamic obstacles
- Conservative behavior: "don't move unless necessary to survive"
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Any
from enum import Enum
import pygame

import gymnasium as gym
from gymnasium import spaces


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
    safe_distance: float = 2.0  # meters, safe clearance distance

    # Reward weights
    collision_penalty: float = -1000.0
    survival_reward: float = 1.0
    movement_penalty_scale: float = 1.0
    action_smoothness_scale: float = 1.0
    threat_passage_reward: float = 10.0
    conservative_bonus_scale: float = 0.3


class SafeStopEnv(gym.Env):
    """
    Conservative lateral avoidance environment following the Gymnasium API.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Optional[BaseEnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or BaseEnvConfig()
        self.render_mode = render_mode
        self.renderer = None

        # 14-dim observation:
        # [0-1]: ego pos (x, y), [2]: ego vy, [3-4]: rel obj pos (dx, dy),
        # [5-6]: rel obj vel (dvx, dvy), [7]: distance, [8]: TTCA, [9]: d_min,
        # [10]: threat flag, [11]: FOV flag, [12]: norm time, [13]: prev action
        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, shape=(14,), dtype=np.float32
        )

        # 1-dim action: lateral velocity target
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

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

        self._last_reward_components = {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset ego
        self.ego_pos = np.array(self.cfg.ego_start_pos, dtype=np.float32)
        self.ego_vy = 0.0
        self.ego_vy_target = 0.0
        self.prev_action = 0.0

        # Spawn object
        x = self.np_random.uniform(*self.cfg.obj_spawn_x_range)
        y = self.np_random.uniform(*self.cfg.obj_spawn_y_range)
        self.obj_pos = np.array([x, y], dtype=np.float32)

        vx = self.np_random.uniform(*self.cfg.obj_vx_range)
        vy = self.np_random.uniform(*self.cfg.obj_vy_range)
        self.obj_vel = np.array([vx, vy], dtype=np.float32)

        # Reset tracking
        self.step_count = 0
        self.threat_passed = False
        self.cooldown_counter = 0
        self.collision_occurred = False
        self.termination_reason = TerminationReason.NONE
        self.threat_active = False

        self._update_threat_state()

        if self.render_mode == "human" and self.renderer is None:
            self.renderer = EnvRenderer(self, scale=20.0)

        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        # Unpack action from array
        action_val = float(np.clip(action[0], -1.0, 1.0))
        self.ego_vy_target = action_val * self.cfg.ego_vy_max

        # Velocity dynamics with acceleration limits
        max_delta_v = self.cfg.ego_max_acceleration * self.cfg.dt
        velocity_error = self.ego_vy_target - self.ego_vy
        velocity_change = np.clip(velocity_error, -max_delta_v, max_delta_v)
        self.ego_vy += velocity_change

        # Update positions
        self.ego_pos[1] += self.ego_vy * self.cfg.dt
        self.obj_pos += self.obj_vel * self.cfg.dt

        self.step_count += 1

        # State updates
        self.collision_occurred = self._check_collision()
        self._update_threat_state()

        # Check termination & truncation
        terminated, truncated = self._check_termination(self.collision_occurred)

        # Reward
        reward = self._compute_reward(action_val, self.collision_occurred)

        self.prev_action = action_val

        # Render if human mode
        if self.render_mode == "human" and self.renderer:
            self.renderer.render(reward=reward, action=action_val)
            self.renderer.tick()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _check_collision(self) -> bool:
        distance = np.linalg.norm(self.obj_pos - self.ego_pos)
        return bool(distance < (self.cfg.ego_radius + self.cfg.obj_radius))

    def _is_object_in_fov(self) -> bool:
        rel_pos = self.obj_pos - self.ego_pos
        distance = np.linalg.norm(rel_pos)
        if distance > self.cfg.fov_range: return False
        if distance < 1e-6: return True
        angle = math.atan2(abs(rel_pos[1]), rel_pos[0])
        half_fov_rad = math.radians(self.cfg.fov_angle / 2)
        return rel_pos[0] > -self.cfg.ego_radius and angle <= half_fov_rad

    def _compute_ttca_and_dmin(self) -> Tuple[float, float]:
        rel_pos = self.obj_pos - self.ego_pos
        rel_vel = self.obj_vel - np.array([0.0, self.ego_vy])
        vel_sq = np.dot(rel_vel, rel_vel)
        if vel_sq < 1e-6:
            return float('inf'), float(np.linalg.norm(rel_pos))
        ttca = max(-np.dot(rel_pos, rel_vel) / vel_sq, 0.0)
        closest_pos = rel_pos + rel_vel * ttca
        return float(ttca), float(np.linalg.norm(closest_pos))

    def _update_threat_state(self):
        if not self._is_object_in_fov():
            return
        _, d_min = self._compute_ttca_and_dmin()
        if self.obj_pos[0] < self.ego_pos[0] - 0.5:
            if self.threat_active and not self.threat_passed:
                self.threat_passed = True
            self.threat_active = False
            return
        self.threat_active = (d_min < self.cfg.threat_threshold)

    def _get_observation(self) -> np.ndarray:
        rel_pos = self.obj_pos - self.ego_pos
        rel_vel = self.obj_vel - np.array([0.0, self.ego_vy])
        dist = np.linalg.norm(rel_pos)
        ttca, d_min = self._compute_ttca_and_dmin()

        obs = np.array([
            self.ego_pos[0], self.ego_pos[1], self.ego_vy,
            rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1],
            dist, ttca, d_min,
            float(self.threat_active), float(self._is_object_in_fov()),
            self.step_count / self.cfg.max_episode_steps, self.prev_action
        ], dtype=np.float32)

        return np.clip(obs, -100.0, 100.0)

    def _compute_reward(self, action: float, collision: bool) -> float:
        reward = 0.0
        components = {}

        if collision:
            components['collision'] = self.cfg.collision_penalty
            self._last_reward_components = components
            return self.cfg.collision_penalty

        ttca, d_min = self._compute_ttca_and_dmin()
        speed = abs(self.ego_vy)
        is_still = speed < self.cfg.stillness_threshold

        components['survival'] = self.cfg.survival_reward
        reward += components['survival']

        if d_min > self.cfg.safe_distance and self._is_object_in_fov():
            if is_still:
                still_bonus = 5.0
                components['safe_still'] = still_bonus
                reward += still_bonus
            else:
                unnecessary_movement_penalty = -10.0 * speed
                components['safe_moving'] = unnecessary_movement_penalty
                reward += unnecessary_movement_penalty
        elif d_min < self.cfg.safe_distance and self._is_object_in_fov():
            movement_penalty = -0.5 * speed
            components['movement'] = movement_penalty
            reward += movement_penalty
        else:
            if is_still:
                still_bonus = 3.0
                components['no_obj_still'] = still_bonus
                reward += still_bonus
            else:
                unnecessary_penalty = -5.0 * speed
                components['no_obj_moving'] = unnecessary_penalty
                reward += unnecessary_penalty

        action_change = abs(action - self.prev_action)
        smoothness_penalty = -0.5 * action_change
        components['smoothness'] = smoothness_penalty
        reward += smoothness_penalty

        if self.threat_passed and not self.collision_occurred:
            components['threat_passed'] = self.cfg.threat_passage_reward
            reward += components['threat_passed']
            self.threat_passed = False
        else:
            components['threat_passed'] = 0.0

        self._last_reward_components = components
        return reward

    def _check_termination(self, collision: bool) -> Tuple[bool, bool]:
        terminated = False
        truncated = False

        if collision:
            self.termination_reason = TerminationReason.COLLISION
            terminated = True
        elif (self.ego_pos[1] < self.cfg.arena_y_range[0] or
              self.ego_pos[1] > self.cfg.arena_y_range[1]):
            self.termination_reason = TerminationReason.COLLISION
            terminated = True
        elif self.obj_pos[0] < self.cfg.arena_x_range[0]:
            self.termination_reason = TerminationReason.SUCCESS
            terminated = True
        elif self.step_count >= self.cfg.max_episode_steps:
            self.termination_reason = TerminationReason.TIMEOUT
            truncated = True

        return terminated, truncated

    def _get_info(self) -> Dict[str, Any]:
        ttca, d_min = self._compute_ttca_and_dmin()
        return {
            "collision": self.collision_occurred,
            "threat_active": self.threat_active,
            "threat_passed": self.threat_passed,
            "termination_reason": self.termination_reason.value,
            "ttca": ttca,
            "d_min": d_min,
            "ego_speed": abs(self.ego_vy),
            "reward_components": self._last_reward_components.copy(),
            "is_success": self.termination_reason == TerminationReason.SUCCESS
        }

    def close(self):
        if self.renderer:
            self.renderer.close()


class EnvRenderer:
    """Pygame-based visualization for the environment."""

    def __init__(self, env: SafeStopEnv, width: int = 1200, height: int = 800, scale: float = 20.0):
        pygame.init()
        self.env = env
        self.width = width
        self.height = height
        self.scale = scale
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Safe Stop Environment")
        self.main_surface = pygame.Surface((width, height))
        self.clock = pygame.time.Clock()
        self.offset_x = width // 2
        self.offset_y = height // 2

        self.colors = {
            "background": (20, 20, 25), "grid": (40, 40, 45),
            "ego_safe": (50, 200, 50), "ego_moving": (200, 200, 50),
            "ego_danger": (255, 100, 50), "object": (220, 50, 50),
            "fov": (100, 100, 150, 50), "text": (200, 200, 200),
        }
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        return int(x * self.scale + self.offset_x), int(self.offset_y - y * self.scale)

    def draw_fov(self):
        cfg = self.env.cfg
        ego_screen = self.world_to_screen(*self.env.ego_pos)
        half_fov_rad = math.radians(cfg.fov_angle / 2)
        points = [ego_screen]
        for i in range(21):
            angle = -half_fov_rad + (2 * half_fov_rad) * i / 20
            end_x = self.env.ego_pos[0] + cfg.fov_range * math.cos(angle)
            end_y = self.env.ego_pos[1] + cfg.fov_range * math.sin(angle)
            points.append(self.world_to_screen(end_x, end_y))
        if len(points) >= 3:
            pygame.draw.polygon(self.main_surface, self.colors["fov"], points)

    def draw_grid(self):
        cfg = self.env.cfg
        for x in range(int(cfg.arena_x_range[0]), int(cfg.arena_x_range[1]) + 1, 5):
            start = self.world_to_screen(x, cfg.arena_y_range[0])
            end = self.world_to_screen(x, cfg.arena_y_range[1])
            pygame.draw.line(self.main_surface, self.colors["grid"], start, end, 1)
        for y in range(int(cfg.arena_y_range[0]), int(cfg.arena_y_range[1]) + 1, 5):
            start = self.world_to_screen(cfg.arena_x_range[0], y)
            end = self.world_to_screen(cfg.arena_x_range[1], y)
            pygame.draw.line(self.main_surface, self.colors["grid"], start, end, 1)

    def draw_ego(self):
        cfg = self.env.cfg
        pos = self.world_to_screen(*self.env.ego_pos)
        radius = int(cfg.ego_radius * self.scale)

        if self.env.collision_occurred:
            color = (255, 50, 50) if int(self.env.step_count * 10) % 2 else (255, 255, 255)
        elif self.env.threat_active:
            color = self.colors["ego_danger"]
        elif abs(self.env.ego_vy) > cfg.stillness_threshold:
            color = self.colors["ego_moving"]
        else:
            color = self.colors["ego_safe"]

        pygame.draw.circle(self.main_surface, color, pos, radius)
        pygame.draw.circle(self.main_surface, (255, 255, 255), pos, radius, 2)

    def draw_object(self):
        cfg = self.env.cfg
        pos = self.world_to_screen(*self.env.obj_pos)
        radius = int(cfg.obj_radius * self.scale)
        in_fov = self.env._is_object_in_fov()
        color = self.colors["object"] if in_fov else (100, 50, 50)

        pygame.draw.circle(self.main_surface, color, pos, radius)
        if in_fov:
            pygame.draw.circle(self.main_surface, (255, 100, 100), pos, radius, 2)

    def draw_info(self, reward: float = 0.0, action: float = 0.0):
        y_offset, line_height = 10, 25
        texts = [
            f"Step: {self.env.step_count}/{self.env.cfg.max_episode_steps}",
            f"Reward: {reward:.2f}",
            f"Action: {action:.3f}",
            f"Ego vy: {self.env.ego_vy:.3f} m/s",
            f"Threat: {'ACTIVE' if self.env.threat_active else 'None'}",
            f"Collision: {'COLLISION!' if self.env.collision_occurred else 'Safe'}",
            f"Status: {self.env.termination_reason.value}",
        ]
        for i, text in enumerate(texts):
            col = (255, 100, 100) if "COLLISION!" in text else self.colors["text"]
            self.main_surface.blit(self.font.render(text, True, col), (10, y_offset + i * line_height))

        if hasattr(self.env, '_last_reward_components') and self.env._last_reward_components:
            y_offset, x_offset = 10, self.width - 280
            self.main_surface.blit(self.font.render("Reward Components:", True, self.colors["text"]), (x_offset, y_offset))
            y_offset += line_height
            for key, value in self.env._last_reward_components.items():
                self.main_surface.blit(self.small_font.render(f"{key}: {value:.3f}", True, self.colors["text"]), (x_offset, y_offset))
                y_offset += 20

    def render(self, reward: float = 0.0, action: float = 0.0):
        self.main_surface.fill(self.colors["background"])
        self.draw_grid()
        self.draw_fov()
        self.draw_object()
        self.draw_ego()
        self.draw_info(reward, action)
        self.screen.blit(self.main_surface, (0, 0))
        pygame.display.flip()

    def tick(self, fps: int = 30):
        self.clock.tick(fps)
        pygame.event.pump() # Prevent Window Freezing

    def close(self):
        pygame.quit()