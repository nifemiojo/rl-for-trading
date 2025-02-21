import random

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration
        self.min_epsilon = min_epsilon  # Minimum exploration
        self.q_table = self._initialize_q_table()

    def _initialize_q_table(self):
        """Initialize Q-table with zeros for all state-action pairs"""
        state_space = self.env.get_state_space()
        action_space = self.env.get_action_space()
        return {state: {action: 0 for action in action_space} for state in state_space}

    def choose_action(self, state):
        """Select action using epsilon-greedy strategy"""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.get_action_space())  # Explore
        else:
            return max(self.q_table[state], key=self.q_table[state].get)  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using the Q-learning formula"""
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        target = reward + self.gamma * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    def train(self, episodes=500):
        """Train the agent using Q-Learning"""
        rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)  # Decay epsilon
            rewards.append(total_reward)
            
            if episode % 50 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")
            """ if episode == 499:
                print(f"Q-Table: {self.q_table}") """

        return rewards
