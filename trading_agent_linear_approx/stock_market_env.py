import numpy as np

class StockMarketEnv:
    def __init__(self, prices: np.ndarray, window_size=5, initial_balance=1000):
        self.prices = prices
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the environment to the initial state"""
        self.current_step = self.window_size - 1
        self.current_balance = self.initial_balance
        self.position = None
        self.entry_price = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray[float]:
        """Return the last `window_size` closing prices as the state"""
        window_prices = self.prices[self.current_step - self.window_size + 1 : self.current_step + 1]

        price_changes_pct = (window_prices[1:] / window_prices[:-1]) - 1

        sma = np.mean(window_prices)

        volatility = np.std(window_prices)

        return np.array([price_changes_pct[-1], sma, volatility])
    
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
                profit = current_price - self.entry_price
                reward = profit * 100 # Scale reward for better learning
                self.current_balance += reward
                self.position = None # Exit position
        
        else: # Hold
            reward = -1 # Small penalty to discourage excessive holding
        
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1 # End of price data
        
        return self._get_state(), reward, done


    def get_action_space(self) -> list[int]:
        """Return possible actions: 0=Sell, 1=Hold, 2=Buy"""
        return [0, 1, 2]
