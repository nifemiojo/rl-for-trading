from stock_market_env import StockMarketEnv
import numpy as np

class LinearQTradingAgent:
    def __init__(self, env: StockMarketEnv, state_size: int, action_size: int, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 1.0, epsilon_decay: float = 0.995, min_epsilon: float = 0.01):
        self.env = env
        self.state_size = state_size  # Number of features
        self.action_size = action_size # Number of possible actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.weights = np.random.uniform(-0.1, 0.1, (action_size, state_size))
        self.weight_changes = []

    def _get_q_value(self, state, action) -> float:
        """Compute Q-value using linear function approximation"""
        return np.dot(self.weights[action], state)

    def choose_action(self, state) -> int:
        """Choose action using epsilon-greedy strategy"""
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size if self.env.position == "buy" else [1, 2])
        else:
            q_values = { action: self._get_q_value(state, action) for action in range(self.action_size) }
            if self.env.position != "buy":
                q_values.pop(0)
            return max(q_values, key=q_values.get)
    
    def update_weights(self, state, action: int, reward: float, next_state):
        """Update weights using gradient descent on the Bellman equation"""
        next_q_values = [self._get_q_value(next_state, a) for a in range(self.action_size)]
        target = reward + self.gamma * max(next_q_values)

        error = target - self._get_q_value(state, action)
        self.weights[action] += self.alpha * error * state
    
    def train(self, episodes=1000):
        """Train the Q-learning agent"""
        rewards: list[float] = []

        for episode in range(episodes):
            state = self.env.reset()
            total_reward: float = 0
            done = False
            old_weights = self.weights.copy()

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_weights(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            self.weight_changes.append(np.mean(np.abs(self.weights - old_weights)))

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            rewards.append(total_reward)

            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")
        
        return rewards