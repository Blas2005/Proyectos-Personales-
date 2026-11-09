import json
import os
import numpy as np


class QLearningAgent:
    """
    Agente de Inteligencia Artificial basado en Q-Learning Tabular con Dyna-Q opcional
    (memoria de experiencia para adaptarse más rápido cuando el usuario mueve la meta o paredes).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.25,
        discount_factor: float = 0.96,
        epsilon: float = 0.25,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.02,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.auto_decay = True

        # Tabla Q: [estado, acción]
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)

        # Estadísticas
        self.episodes = 0
        self.total_steps = 0
        self.wins = 0
        self.reward_history = []
        self.steps_history = []

    def select_action(self, state: int, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        # Desempate aleatorio entre acciones con el mismo valor máximo
        q_vals = self.q_table[state]
        max_q = np.max(q_vals)
        best_actions = np.flatnonzero(np.isclose(q_vals, max_q))
        return int(np.random.choice(best_actions))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        best_next_q = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        self.total_steps += 1

    def end_episode(self, episode_reward: float, episode_steps: int, won: bool):
        self.episodes += 1
        if won:
            self.wins += 1
        self.reward_history.append(float(episode_reward))
        self.steps_history.append(int(episode_steps))
        if len(self.reward_history) > 200:
            self.reward_history.pop(0)
            self.steps_history.pop(0)

        if self.auto_decay and self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_brain(self):
        self.q_table.fill(0.0)
        self.episodes = 0
        self.total_steps = 0
        self.wins = 0
        self.reward_history.clear()
        self.steps_history.clear()
        self.epsilon = 0.30

    def save_brain(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = {
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "episodes": self.episodes,
            "total_steps": self.total_steps,
            "wins": self.wins,
            "reward_history": self.reward_history,
            "q_table": self.q_table.tolist(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_brain(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("n_states") == self.n_states and data.get("n_actions") == self.n_actions:
            self.q_table = np.array(data["q_table"], dtype=np.float32)
            self.lr = float(data.get("lr", self.lr))
            self.gamma = float(data.get("gamma", self.gamma))
            self.epsilon = float(data.get("epsilon", self.epsilon))
            self.episodes = int(data.get("episodes", 0))
            self.total_steps = int(data.get("total_steps", 0))
            self.wins = int(data.get("wins", 0))
            self.reward_history = list(data.get("reward_history", []))
            return True
        return False
