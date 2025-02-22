class Grid:
    def __init__(self, size=4, goal=(3, 3), penalty=(3, 2)):
        self.size = size
        self.goal = goal
        self.penalty = penalty
        self.reset()
    
    def reset(self):
        """Reset agent to the start position"""
        self.agent_pos = (0, 0)
        return self.agent_pos
    
    def step(self, action):
        """Apply an action and return new state, reward, and done flag"""
        x, y = self.agent_pos

        if action == 0:
            # Move up
            x = max(x - 1, 0)
        elif action == 1:
            # Move down
            x = min(x + 1, self.size - 1)
        elif action == 2:
             # Move left
             y = max(y - 1, 0)
        elif action == 3:
            # Move right
            y = min(y + 1, self.size - 1)
    
        self.agent_pos = (x, y)

        if self.agent_pos == self.goal:
            return self.agent_pos, 10, True  # Reward for reaching goal
        elif self.agent_pos == self.penalty:
            return self.agent_pos, -10, True  # Penalty for wrong move
        else:
            return self.agent_pos, -1, False  # Small penalty for each move
    
    def get_state_space(self):
        """Return all possible states"""
        return [(i, j) for i in range(self.size) for j in range(self.size)]
    
    def get_action_space(self):
        """Return action space: 0=Up, 1=Down, 2=Left, 3=Right"""
        return [0, 1, 2, 3]