import numpy as np
from stock_market_env import StockMarketEnv
from trading_agent import LinearQTradingAgent

# Simulated stock prices
prices = np.random.default_rng().normal(100, 1.5, 1000)
price_changes_pct = (prices[1:] / prices[:-1]) - 1

# Create the environment
env = StockMarketEnv(prices, window_size=5)

# Train the agent
state_size = 3  # 3 features: return, SMA, volatility
action_size = 3  # Actions: Buy, Hold, Sell
agent = LinearQTradingAgent(env, state_size, action_size)
rewards = agent.train()

print(agent.weights)