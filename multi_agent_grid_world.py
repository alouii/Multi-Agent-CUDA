import cupy as cp
import numpy as np
import time

# Configuration
GRID_SIZE = 100  # 100x100 grid
NUM_AGENTS = 1000
NUM_STEPS = 100
SEED = 42

# Set random seed for reproducibility
cp.random.seed(SEED)
np.random.seed(SEED)

def initialize_agents(num_agents, grid_size):
    """Initialize agent positions randomly on the grid."""
    # Randomly assign x, y coordinates
    agent_positions = cp.zeros((num_agents, 2), dtype=cp.int32)
    agent_positions[:, 0] = cp.random.randint(0, grid_size, num_agents)  # x-coordinates
    agent_positions[:, 1] = cp.random.randint(0, grid_size, num_agents)  # y-coordinates
    return agent_positions

@cp.fusion.fuse()
def update_positions(agent_positions, grid_size, directions):
    """Update agent positions based on random directions."""
    new_positions = agent_positions.copy()
    # Directions: 0=up, 1=down, 2=left, 3=right
    new_positions[:, 1] += cp.where(directions == 0, -1, 0)  # Up
    new_positions[:, 1] += cp.where(directions == 1, 1, 0)   # Down
    new_positions[:, 0] += cp.where(directions == 2, -1, 0)  # Left
    new_positions[:, 0] += cp.where(directions == 3, 1, 0)   # Right
    # Ensure positions stay within grid bounds
    new_positions[:, 0] = cp.clip(new_positions[:, 0], 0, grid_size - 1)
    new_positions[:, 1] = cp.clip(new_positions[:, 1], 0, grid_size - 1)
    return new_positions

def check_collisions(new_positions, agent_positions):
    """Check for collisions and revert moves if target cell is occupied."""
    # Create a grid to track occupied cells
    grid = cp.zeros((GRID_SIZE, GRID_SIZE), dtype=cp.int32)
    # Mark current positions as occupied (agent index + 1 to distinguish from empty)
    for i in range(NUM_AGENTS):
        x, y = agent_positions[i]
        grid[x, y] = i + 1
    
    # Check for collisions
    valid_moves = cp.ones(NUM_AGENTS, dtype=cp.bool_)
    for i in range(NUM_AGENTS):
        new_x, new_y = new_positions[i]
        # If target cell is occupied (non-zero) and not by the same agent, mark move as invalid
        if grid[new_x, new_y] != 0 and grid[new_x, new_y] != i + 1:
            valid_moves[i] = False
    
    # Update positions only for valid moves
    final_positions = cp.where(valid_moves[:, None], new_positions, agent_positions)
    
    return final_positions

def main():
    # Initialize agents
    agent_positions = initialize_agents(NUM_AGENTS, GRID_SIZE)
    print(f"Initialized {NUM_AGENTS} agents on a {GRID_SIZE}x{GRID_SIZE} grid.")
    
    # Simulation loop
    start_time = time.time()
    for step in range(NUM_STEPS):
        # Generate random directions (0=up, 1=down, 2=left, 3=right)
        directions = cp.random.randint(0, 4, NUM_AGENTS)
        
        # Update positions
        new_positions = update_positions(agent_positions, GRID_SIZE, directions)
        
        # Check for collisions and finalize positions
        agent_positions = check_collisions(new_positions, agent_positions)
        
        # Optional: Print progress every 10 steps
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{NUM_STEPS} completed.")
    
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    
    # Transfer final positions to CPU for inspection (optional)
    final_positions = cp.asnumpy(agent_positions)
    print(f"Final positions of first 5 agents:\n{final_positions[:5]}")

if __name__ == "__main__":
    try:
        main()
    except cp.cuda.memory.OutOfMemoryError:
        print("CUDA out of memory. Try reducing NUM_AGENTS or GRID_SIZE.")
