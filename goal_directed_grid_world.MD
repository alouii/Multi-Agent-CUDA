# Goal-Directed Multi-Agent Grid World Simulation with CUDA

This project implements a multi-agent grid world where 1,000 agents navigate toward individual goal positions on a 100x100 grid, with CUDA acceleration using CuPy. Agents move to minimize their Manhattan distance to their goals, avoiding collisions with other agents. The simulation is optimized for GPU performance, making it suitable for large-scale agent-based modeling.

#Overview

Environment: A 100x100 2D grid where agents move toward unique goal positions.
Agents: 1,000 agents, each with a random starting position and goal.
Behavior: Agents move (up, down, left, right) to minimize Manhattan distance to their goal, staying in place if the target cell is occupied.
CUDA Acceleration: CuPy parallelizes movement, collision detection, and goal-checking on the GPU.
Simulation: Runs until all agents reach their goals or for a maximum of 1,000 steps.

# Features

Random initialization of agent and goal positions.
Goal-directed movement based on Manhattan distance.
Collision avoidance to prevent overlapping agents.
GPU-accelerated updates using CuPy with kernel fusion.
Tracks and reports the number of agents reaching their goals.

# Requirements

Hardware: CUDA-enabled GPU with sufficient memory.
Software:Python 3.8+
CuPy
NumPy

CUDA Toolkit: Ensure compatibility with CuPy (see CuPy installation guide).

# Installation

install dependencies:
'''bash
pip install cupy-cuda12x numpy


'''
Verify GPU and CUDA setup with nvidia-smi.

UsageSave the script as goal_directed_grid_world.py.
Run the simulation:bash

python goal_directed_grid_world.py 

The script will:Initialize 1,000 agents and goals.
Run until all agents reach their goals or for 1,000 steps.
Print progress every 100 steps, including the number of agents at their goals.
Output final stats and positions.

# Configuration

Edit goal_directed_grid_world.py to modify:GRID_SIZE: Grid size (default: 100).
NUM_AGENTS: Number of agents (default: 1,000).
MAX_STEPS: Maximum simulation steps (default: 1,000).
SEED: Random seed (default: 42).

Example Output

Initialized 1000 agents with goals on a 100x100 grid.
Step 100/1000: 250 agents reached their goals.
Step 200/1000: 600 agents reached their goals.
...
All agents reached their goals at step 350.
Simulation completed in 0.22 seconds.
Final stats: 1000 agents reached their goals.
Final positions of first 5 agents:
[[10 20]
 [45 67]
 [23 89]
 [12 34]
 [78 56]]
Goal positions of first 5 agents:
[[10 20]
 [45 67]
 [23 89]
 [12 34]
 [78 56]]

 # Implementation 
 
 DetailsInitialization: Random agent and goal positions using CuPy.
Movement: Agents choose the direction minimizing Manhattan distance to their goal, with boundary clipping.
Collision Detection: Grid-based occupancy check; invalid moves keep agents in place.
Goal Tracking: Monitors agents reaching their goals (Manhattan distance = 0).
CUDA: CuPy with @cp.fusion.fuse() optimizes GPU performance.

# Limitations

Deadlocks: Dense scenarios may cause agents to block each other. Consider priority rules or pathfinding.
Visualization: Excluded for simplicity. Add Matplotlib for visual output (use cp.asnumpy()).
Behavior: Simple Manhattan distance-based movement. Extend with A* or other algorithms for smarter navigation.

# Potential Extensions

Add obstacles or dynamic goals.
Implement advanced pathfinding (e.g., A*).
Visualize agent paths or grid state.
Export simulation data for analysis.

# Troubleshooting

CUDA Out of Memory: Reduce NUM_AGENTS or GRID_SIZE.
CuPy Errors: Verify CuPy and CUDA compatibility.
Slow Performance: Ensure GPU execution (CuPy) and check kernel fusion.

# License

Provided as-is for educational and research purposes.AcknowledgmentsBuilt with CuPy for GPU acceleration.
Inspired by goal-directed multi-agent systems.


