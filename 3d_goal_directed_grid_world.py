import cupy as cp
import numpy as np
import time

# Configuration
GRID_SIZE = 50  # 50x50x50 grid
NUM_AGENTS = 1000
MAX_STEPS = 1000
SEED = 42

# Set random seed for reproducibility
cp.random.seed(SEED)
np.random.seed(SEED)

def initialize_agents_and_goals(num_agents, grid_size):
    """Initialize agent positions and goal positions randomly in 3D grid."""
    # Agent positions: [x, y, z]
    agent_positions = cp.zeros((num_agents, 3), dtype=cp.int32)
    agent_positions[:, 0] = cp.random.randint(0, grid_size, num_agents)  # x
    agent_positions[:, 1] = cp.random.randint(0, grid_size, num_agents)  # y
    agent_positions[:, 2] = cp.random.randint(0, grid_size, num_agents)  # z
    
    # Goal positions: [x, y, z]
    goal_positions = cp.zeros((num_agents, 3), dtype=cp.int32)
    goal_positions[:, 0] = cp.random.randint(0, grid_size, num_agents)
    goal_positions[:, 1] = cp.random.randint(0, grid_size, num_agents)
    goal_positions[:, 2] = cp.random.randint(0, grid_size, num_agents)
    
    return agent_positions, goal_positions

@cp.fusion.fuse()
def compute_manhattan_distances(agent_positions, goal_positions):
    """Compute Manhattan distance from each agent to its goal in 3D."""
    return cp.abs(agent_positions[:, 0] - goal_positions[:, 0]) + \
           cp.abs(agent_positions[:, 1] - goal_positions[:, 1]) + \
           cp.abs(agent_positions[:, 2] - goal_positions[:, 2])

@cp.fusion.fuse()
def choose_directions(agent_positions, goal_positions, grid_size):
    """Choose direction to minimize Manhattan distance to goal in 3D."""
    directions = cp.zeros(NUM_AGENTS, dtype=cp.int32)  # 0=up, 1=down, 2=left, 3=right, 4=forward, 5=backward
    
    # Compute possible new positions for 6 directions
    new_positions = cp.zeros((NUM_AGENTS, 6, 3), dtype=cp.int32)  # [agent, direction, (x,y,z)]
    
    # Up (y-1)
    new_positions[:, 0] = agent_positions
    new_positions[:, 0, 1] -= 1
    # Down (y+1)
    new_positions[:, 1] = agent_positions
    new_positions[:, 1, 1] += 1
    # Left (x-1)
    new_positions[:, 2] = agent_positions
    new_positions[:, 2, 0] -= 1
    # Right (x+1)
    new_positions[:, 3] = agent_positions
    new_positions[:, 3, 0] += 1
    # Forward (z-1)
    new_positions[:, 4] = agent_positions
    new_positions[:, 4, 2] -= 1
    # Backward (z+1)
    new_positions[:, 5] = agent_positions
    new_positions[:, 5, 2] += 1
    
    # Clip to grid boundaries
    new_positions[:, :, 0] = cp.clip(new_positions[:, :, 0], 0, grid_size - 1)
    new_positions[:, :, 1] = cp.clip(new_positions[:, :, 1], 0, grid_size - 1)
    new_positions[:, :, 2] = cp.clip(new_positions[:, :, 2], 0, grid_size - 1)
    
    # Compute Manhattan distances for each direction
    distances = cp.zeros((NUM_AGENTS, 6), dtype=cp.int32)
    for d in range(6):
        distances[:, d] = cp.abs(new_positions[:, d, 0] - goal_positions[:, 0]) + \
                          cp.abs(new_positions[:, d, 1] - goal_positions[:, 1]) + \
                          cp.abs(new_positions[:, d, 2] - goal_positions[:, 2])
    
    # Choose direction with minimum distance
    directions = cp.argmin(distances, axis=1)
    
    return directions, new_positions

def check_collisions_and_update(agent_positions, new_positions, directions):
    """Check for collisions and update positions for valid moves."""
    # Create 3D grid to track occupied cells
    grid = cp.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=cp.int32)
    for i in range(NUM_AGENTS):
        x, y, z = agent_positions[i]
        grid[x, y, z] = i + 1  # Agent index + 1 (0 is empty)
    
    # Check for collisions
    valid_moves = cp.ones(NUM_AGENTS, dtype=cp.bool_)
    for i in range(NUM_AGENTS):
        d = directions[i]
        new_x, new_y, new_z = new_positions[i, d]
        if grid[new_x, new_y, new_z] != 0 and grid[new_x, new_y, new_z] != i + 1:
            valid_moves[i] = False
    
    # Update positions for valid moves
    final_positions = cp.zeros_like(agent_positions)
    for i in range(NUM_AGENTS):
        if valid_moves[i]:
            final_positions[i] = new_positions[i, directions[i]]
        else:
            final_positions[i] = agent_positions[i]
    
    return final_positions

def main():
    # Initialize agents and goals
    agent_positions, goal_positions = initialize_agents_and_goals(NUM_AGENTS, GRID_SIZE)
    print(f"Initialized {NUM_AGENTS} agents with goals on a {GRID_SIZE}x{GRID_SIZE}x{GRID_SIZE} 3D grid.")
    
    # Track agents that have reached their goals
    reached_goal = cp.zeros(NUM_AGENTS, dtype=cp.bool_)
    start_time = time.time()
    
    # Simulation loop
    for step in range(MAX_STEPS):
        # Check which agents have reached their goals
        distances = compute_manhattan_distances(agent_positions, goal_positions)
        reached_goal = distances == 0
        num_reached = cp.sum(reached_goal)
        
        # Stop if all agents have reached their goals
        if num_reached == NUM_AGENTS:
            print(f"All agents reached their goals at step {step + 1}.")
            break
        
        # Choose directions to minimize Manhattan distance
        directions, new_positions = choose_directions(agent_positions, goal_positions, GRID_SIZE)
        
        # Update positions with collision checking
        agent_positions = check_collisions_and_update(agent_positions, new_positions, directions)
        
        # Print progress every 100 steps
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{MAX_STEPS}: {num_reached} agents reached their goals.")
    
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    
    # Final stats
    final_distances = cp.asnumpy(compute_manhattan_distances(agent_positions, goal_positions))
    print(f"Final stats: {np.sum(final_distances == 0)} agents reached their goals.")
    print(f"Final positions of first 5 agents:\n{cp.asnumpy(agent_positions[:5])}")
    print(f"Goal positions of first 5 agents:\n{cp.asnumpy(goal_positions[:5])}")

if __name__ == "__main__":
    try:
        main()
    except cp.cuda.memory.OutOfMemoryError:
        print("CUDA out of memory. Try reducing NUM_AGENTS or GRID_SIZE.")