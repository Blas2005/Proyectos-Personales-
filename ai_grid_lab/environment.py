import json
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class GridWorldEnv(gym.Env):
    """
    Entorno liviano de cuadrícula (Grid World) compatible con Gymnasium.
    Permite editar paredes, trampas, inicio y meta en vivo.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Tipos de celda
    EMPTY = 0
    WALL = 1
    TRAP = 2
    START = 3
    GOAL = 4

    # Acciones
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    ACTION_DELTAS = {
        UP: (-1, 0),
        RIGHT: (0, 1),
        DOWN: (1, 0),
        LEFT: (0, -1),
    }

    def __init__(self, rows: int = 12, cols: int = 16, max_steps: int = 150):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(rows * cols)

        self.grid = np.zeros((rows, cols), dtype=np.int32)
        self.start_pos = (1, 1)
        self.goal_pos = (rows - 2, cols - 2)
        self.agent_pos = self.start_pos
        self.steps_taken = 0
        self.trail = []

        # Recompensas configurables
        self.reward_goal = 100.0
        self.reward_trap = -50.0
        self.reward_wall = -2.0
        self.reward_step = -0.2

        self.load_default_maze()

    def load_default_maze(self):
        """Carga un mapa inicial con algunos obstáculos y trampas."""
        self.grid.fill(self.EMPTY)
        self.start_pos = (1, 1)
        self.goal_pos = (self.rows - 2, self.cols - 2)

        # Paredes de ejemplo
        for r in range(0, self.rows - 3):
            self.grid[r, 5] = self.WALL
        for r in range(3, self.rows):
            self.grid[r, 10] = self.WALL

        # Algunas trampas (fuego/pozos)
        self.grid[4, 7] = self.TRAP
        self.grid[5, 7] = self.TRAP
        self.grid[7, 13] = self.TRAP
        self.grid[8, 13] = self.TRAP

        # Asegurar que inicio y meta estén libres
        self.grid[self.start_pos] = self.EMPTY
        self.grid[self.goal_pos] = self.EMPTY
        self.reset()

    def clear_map(self):
        """Limpia todas las paredes y trampas."""
        self.grid.fill(self.EMPTY)
        self.reset()

    def pos_to_state(self, pos: tuple[int, int]) -> int:
        return pos[0] * self.cols + pos[1]

    def state_to_pos(self, state: int) -> tuple[int, int]:
        return divmod(state, self.cols)

    def set_cell(self, r: int, c: int, cell_type: int):
        """Modifica una celda del mapa en tiempo real."""
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return

        if cell_type == self.START:
            if (r, c) != self.goal_pos:
                self.grid[r, c] = self.EMPTY
                self.start_pos = (r, c)
                self.agent_pos = self.start_pos
        elif cell_type == self.GOAL:
            if (r, c) != self.start_pos:
                self.grid[r, c] = self.EMPTY
                self.goal_pos = (r, c)
        else:
            if (r, c) != self.start_pos and (r, c) != self.goal_pos:
                self.grid[r, c] = cell_type

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self.steps_taken = 0
        self.trail = [self.agent_pos]
        return self.pos_to_state(self.agent_pos), {}

    def step(self, action: int):
        self.steps_taken += 1
        dr, dc = self.ACTION_DELTAS[int(action)]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        reward = self.reward_step
        terminated = False
        truncated = False

        # Verificar límites del mapa o pared
        if not (0 <= nr < self.rows and 0 <= nc < self.cols) or self.grid[nr, nc] == self.WALL:
            reward += self.reward_wall
            nr, nc = self.agent_pos  # Rebota y se queda en el lugar
        else:
            self.agent_pos = (nr, nc)

        if len(self.trail) == 0 or self.trail[-1] != self.agent_pos:
            self.trail.append(self.agent_pos)
            if len(self.trail) > 40:
                self.trail.pop(0)

        # Verificar si llegó a la meta o cayó en una trampa
        if self.agent_pos == self.goal_pos:
            reward = self.reward_goal
            terminated = True
        elif self.grid[self.agent_pos] == self.TRAP:
            reward = self.reward_trap
            terminated = True

        if self.steps_taken >= self.max_steps:
            truncated = True

        return self.pos_to_state(self.agent_pos), reward, terminated, truncated, {}

    def save_map(self, filepath: str):
        data = {
            "rows": self.rows,
            "cols": self.cols,
            "start_pos": list(self.start_pos),
            "goal_pos": list(self.goal_pos),
            "grid": self.grid.tolist(),
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_map(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("rows") == self.rows and data.get("cols") == self.cols:
            self.grid = np.array(data["grid"], dtype=np.int32)
            self.start_pos = tuple(data["start_pos"])
            self.goal_pos = tuple(data["goal_pos"])
            self.reset()
            return True
        return False
