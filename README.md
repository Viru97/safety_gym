# 🛑 Safe Stop: Conservative Obstacle Avoidance via Reinforcement Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.0-brightgreen.svg)](https://gymnasium.farama.org/)
[![Stable Baselines3](https://img.shields.io/badge/Stable%20Baselines3-2.8.0-purple.svg)](https://stable-baselines3.readthedocs.io/en/master/)
[![Pygame-CE](https://img.shields.io/badge/Pygame--CE-2.4+-yellow.svg)](https://pyga.me/)

A custom reinforcement learning project where an ego-robot learns to avoid high-speed dynamic obstacles. 

Unlike standard RL agents that tend to exhibit erratic, "jittery" behavior, the core philosophy of this environment is **Conservative Movement**. The agent is explicitly rewarded for recognizing when an obstacle will pass safely and choosing to remain completely still, moving *only* when strictly necessary for survival.

---

## 📸 Demo

*(Add a GIF here showing your trained agent successfully dodging a complex obstacle)*
> **Figure 1:** The agent (green) detects an incoming threat (red), calculates the minimum distance at the closest point of approach (`d_min`), and executes a minimal-movement dodge before returning to a complete stop.

*(Add a second GIF showing the agent staying still when an obstacle passes safely)*
> **Figure 2:** The agent correctly identifies that the object's trajectory poses no threat and remains perfectly still, maximizing its conservative reward.

---

## 🧠 Core Philosophy & Reward Design

In real-world robotics, such as autonomous driving or warehouse automation, unnecessary movement wastes energy, causes mechanical wear, and reduces passenger comfort. 

To achieve "smooth and lazy" behavior, the environment relies on advanced kinematics (Time-To-Closest-Approach, or TTCA) to dynamically scale rewards based on the threat level.

### The Reward Structure
1. **Collision Penalty:** A terminal penalty of `-1000` for failing to avoid an obstacle.
2. **Survival Base:** A small positive reward (`+1.0`) for every step survived.
3. **The Conservative Bonus (The Secret Sauce):**
   * **Safe Scenario (`d_min > safe_distance`):** If the obstacle will naturally miss the robot, the agent receives a massive bonus (`+5.0`) for keeping its velocity near 0 m/s. Any movement during a safe scenario is heavily penalized (`-10.0 * speed`).
   * **Danger Scenario (`d_min < safe_distance`):** If a collision is imminent, the stillness bonus is removed. The agent is allowed to move, but incurs a minor penalty proportional to its speed (`-0.5 * speed`) to encourage the *minimal viable dodge*.
4. **Smoothness:** A penalty based on the difference between the current and previous action to prevent high-frequency oscillations.

---

## 🏗️ Environment Architecture (Gymnasium)

The environment (`SafeStopEnv`) is fully compliant with the modern `Gymnasium` API, allowing seamless integration with Stable Baselines3.

### Observation Space (14-Dimensional Continuous)
The agent perceives its environment through a 14-element vector, bounded between `[-100, 100]`:
* **Ego State:** `[x, y]` position, current lateral velocity `[vy]`.
* **Relative Kinematics:** Relative `[dx, dy]` and `[dvx, dvy]` of the object.
* **Threat Metrics:** Absolute distance, Time-To-Closest-Approach (TTCA), Minimum Distance at CPA (`d_min`).
* **Flags:** Active Threat (boolean), Object in Field of View (boolean).
* **Temporal:** Normalized episode time, previous action.

### Action Space (1-Dimensional Continuous)
* **Target Lateral Velocity:** `[-1.0, 1.0]`. The ego robot controls only its lateral movement (`vy`). The environment applies real-world physics, clipping instantaneous velocity changes to a maximum acceleration limit (`2.0 m/s²`).

---

## 🎓 Curriculum Learning

Training an agent to understand complex spatial threats from scratch is highly inefficient. This project utilizes a custom **Stable Baselines3 Callback** to implement Curriculum Learning. 

As training progresses (measured by total timesteps), the environment seamlessly scales in difficulty:
1. **Stage 1 (Easy):** Objects spawn far away and move slowly (`vx_range: -1.0 to -0.5`).
2. **Stage 2 (Medium):** Object speed increases.
3. **Stage 3 (Hard):** Objects spawn closer and move at maximum speed (`vx_range: -2.0 to -0.5`), requiring immediate, highly precise reaction times.

---

## ⚙️ Installation

It is highly recommended to use Python 3.9, 3.10, or 3.11 to ensure compatibility with pre-built PyTorch and Pygame-CE binaries.

~~~bash
# 1. Clone the repository
git clone https://github.com/yourusername/safety_gym.git
cd safety_gym

# 2. Create a virtual environment (Conda recommended)
conda create -n safe_stop python=3.9
conda activate safe_stop

# 3. Install dependencies
pip install -r requirements.txt
~~~

---

## 🚀 Usage

### Training the Agent
The training script utilizes SB3's `make_vec_env` to parallelize the environment across multiple CPU cores, vastly accelerating data collection.

~~~bash
# Train on 8 parallel environments for 2,000,000 timesteps
python scripts/train.py --n_envs 8 --total_timesteps 2000000
~~~
*Training progress is logged via Tensorboard. To view metrics in real-time, run `tensorboard --logdir logs/` in a separate terminal.*

### Evaluating the Agent
Watch your trained agent perform in real-time using the custom Pygame renderer.

~~~bash
# Renders the environment and loads the best trained model
python scripts/evaluate.py
~~~

---

## 📂 Project Structure

~~~text
safety_gym/
├── envs/
│   ├── __init__.py
│   └── safe_stop_env.py      # Core Gymnasium environment & Pygame renderer
├── scripts/
│   ├── train.py              # PPO Training loop & Curriculum Callback
│   └── evaluate.py           # Evaluation script for visual playback
├── logs/                     # Tensorboard logs & saved models (generated)
├── requirements.txt          # Project dependencies
└── README.md                 # You are here!
~~~

---

## 🔮 Future Enhancements
* **Multi-Obstacle Tracking:** Expanding the observation space to handle `N` simultaneous dynamic obstacles.
* **Recurrent Policies (PPO-LSTM):** Giving the agent memory to better predict trajectories when objects temporarily leave its Field of View (FOV).
* **Continuous Action Masking:** Dynamically restricting the action space when the agent is physically backed against a wall to speed up training convergence.

---
*Developed as part of a Reinforcement Learning research assignment.*