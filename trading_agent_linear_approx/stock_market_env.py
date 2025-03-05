import numpy as np
from sklearn.preprocessing import StandardScaler

class StockMarketEnv:
    def __init__(self, prices: np.ndarray, window_size=5, initial_balance=1000):
        self.scaler = StandardScaler()
        self.prices = prices
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.precomputed_states = []
        self.precompute_scaling()
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the environment to the initial state"""
        self.current_step = self.window_size
        self.current_balance = self.initial_balance
        self.position = None
        self.entry_price = 0
        return self._get_state()

    def _compute_state(self, step) -> np.ndarray[float]:
        """Return the last `window_size` closing prices as the state"""
        window_prices = self.prices[step - self.window_size : step]

        price_changes_pct = (window_prices[1:] / window_prices[:-1]) - 1
        sma = np.mean(window_prices)
        volatility = np.std(window_prices)

        return np.array([price_changes_pct[-1], sma, volatility])
    
    def precompute_scaling(self):
        """Compute mean & std for each feature before training"""
        states = []
        for i in range(self.window_size, len(self.prices)):
            states.append(self._compute_state(i))  # Collect all states

        self.scaler.fit(states)  # Fit scaler on the dataset

        self.precomputed_states = self.scaler.transform(states)

    def _get_state(self):
        """Retrieve current state and apply standardization"""
        return self.precomputed_states[self.current_step - self.window_size]
    
    def step(self, action) -> tuple[np.ndarray, float, bool]:
        """Take an action and return next_state, reward, done"""
        done = False
        reward = 0
        current_price = self.prices[self.current_step]

        if action == 2: # Buy
            if self.position is None:
                self.position = "buy"
                self.entry_price = current_price
        
        elif action == 0: # Sell
            if self.position == "buy": # Only sell if holding a position
                profit = (current_price - self.entry_price) / self.entry_price
                reward = profit * 1.5 # Scale reward for better learning
                self.current_balance += current_price - self.entry_price
                self.position = None # Exit position
        
        else: # Hold
            reward = -0.001 # Small penalty to discourage excessive holding

        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1 # End of price data

        return self._get_state(), reward, done


    def get_action_space(self) -> list[int]:
        """Return possible actions: 0=Sell, 1=Hold, 2=Buy"""
        return [0, 1, 2]
