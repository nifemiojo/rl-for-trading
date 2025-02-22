from stock_market_env import StockMarketEnv
import random
import numpy as np

class TradingAgent:
    def __init__(self, env: StockMarketEnv, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 1.0, epsilon_decay: float = 0.995, min_epsilon: float = 0.01):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.q_table: dict[tuple, dict[int, float]] = {}

    def _get_q_values(self, state: np.ndarray | dict[tuple]):
        """Return Q-values for a given state, initializing if needed"""
        state_tuple = tuple(state)
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = {action: 0 for action in self.env.get_action_space()}
        return self.q_table[state_tuple]

    def choose_action(self, state: np.ndarray | dict[tuple]) -> int:
        """Choose action using epsilon-greedy strategy"""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.get_action_space())
        else:
            q_values = self._get_q_values(state)
            return max(q_values, key=q_values.get)
    
    def update_q_value(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray[float]):
        state_tuple = tuple(state)
        next_state_tuple = tuple(next_state)

        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = {action: 0 for action in self.env.get_action_space()}

        if next_state_tuple not in self.q_table:
            self.q_table[next_state_tuple] = {action: 0 for action in self.env.get_action_space()}
        
        best_next_action = max(self.q_table[next_state_tuple], key=self.q_table[next_state_tuple].get)
        target = reward + self.gamma * self.q_table[next_state_tuple][best_next_action]
        self.q_table[state_tuple][action] += self.alpha * (target - self.q_table[state_tuple][action])
    
    def train(self, episodes=1000):
        """Train the Q-learning agent"""
        rewards: list[float] = []
        for episode in range(episodes):
            state = self.env.reset()
            total_reward: float = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                total_reward += reward
        
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            rewards.append(total_reward)

            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")
        
        return rewards