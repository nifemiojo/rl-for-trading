import matplotlib.pyplot as plt
from q_learning_agent import QLearningAgent
from grid_world import Grid

env = Grid()
agent = QLearningAgent(env)
rewards = agent.train(episodes=500)

""" plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Q-Learning Training Progress")
plt.show() """

def test_agent(agent, env):
    """Run a single test episode with the trained agent"""
    state = env.reset()
    done = False
    path = [state]

    while not done:
        action = max(agent.q_table[state], key=agent.q_table[state].get)
        state, _, done = env.step(action)
        path.append(state)

    return path

# Run the test
test_path = test_agent(agent, env)
print("Optimal Path:", test_path)

