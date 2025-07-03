# 3D Goal-Directed Multi-Agent Grid World Simulation with CUDA


This project implements a 3D multi-agent grid world where 1,000 agents navigate toward individual goal positions in a 50x50x50 grid, accelerated using CUDA via CuPy. Agents move in six directions (up, down, left, right, forward, backward) to minimize their Manhattan distance to their goals, with collision avoidance. The simulation is optimized for GPU performance and stops when all agents reach their goals or after a maximum number of steps.OverviewEnvironment: A 50x50x50 3D grid where agents move toward unique goal positions.
Agents: 1,000 agents, each with a random starting position (x, y, z) and goal.
Behavior: Agents move to minimize 3D Manhattan distance to their goal, staying in place if the target cell is occupied.
CUDA Acceleration: CuPy parallelizes movement, collision detection, and goal-checking.
Simulation: Runs until all agents reach their goals or for 1,000 steps.

# Features

Random initialization of 3D agent and goal positions.
Goal-directed movement based on 3D Manhattan distance.
Collision avoidance in a 3D grid.
GPU-accelerated updates with CuPy and kernel fusion.
Tracks agents reaching their goals and stops early if all succeed.

# Requirements


Hardware: CUDA-enabled GPU with sufficient memory.
Software:Python 3.8+
CuPy
NumPy

CUDA Toolkit: Must match CuPy version (see CuPy installation guide).

# Installation

install dependencies:bash
python 3d_goal_directed_grid_world.py


The script will:Initialize 1,000 agents and goals in a 50x50x50 grid.
Run until all agents reach their goals or for 1,000 steps.
Print progress every 100 steps, including agents at goals.
Output final stats and positions.

# Configuration

Edit 3d_goal_directed_grid_world.py to modify:GRID_SIZE: Grid dimension (default: 50).
NUM_AGENTS: Number of agents (default: 1,000).
MAX_STEPS: Maximum steps (default: 1,000).
SEED: Random seed (default: 42).

Example Output

Initialized 1000 agents with goals on a 50x50x50 3D grid.
Step 100/1000: 200 agents reached their goals.
Step 200/1000: 550 agents reached their goals.
...
All agents reached their goals at step 400.
Simulation completed in 0.28 seconds.
Final stats: 1000 agents reached their goals.
Final positions of first 5 agents:
[[10 20 15]
 [45 30 25]
 [23 40 10]
 [12 15 35]
 [30 25 45]]
Goal positions of first 5 agents:
[[10 20 15]
 [45 30 25]
 [23 40 10]
 [12 15 35]
 [30 25 45]]



# Implementation

 DetailsInitialization: Random 3D positions for agents and goals using CuPy.
Movement: Agents select from six directions to minimize 3D Manhattan distance, with boundary clipping.
Collision Detection: Uses a 3D grid to track occupancy; invalid moves keep agents in place.
Goal Tracking: Monitors agents at goals (Manhattan distance = 0).
CUDA: CuPy with @cp.fusion.fuse() optimizes GPU performance.

# Limitations

Memory: 3D grid (50x50x50) is memory-intensive. Reduce GRID_SIZE if CUDA out-of-memory errors occur.
Deadlocks: Dense scenarios may cause agents to block each other. Consider priority rules or pathfinding.
Visualization: Excluded for simplicity. Add a 3D visualization library (e.g., Matplotlib, Open3D) for output.
Behavior: Simple Manhattan distance movement. Extend with 3D A* or other algorithms.

# Potential Extensions

Add 3D obstacles or dynamic goals.
Implement 3D pathfinding (e.g., A*).
Visualize agent paths in 3D (requires CPU transfer with cp.asnumpy()).
Export simulation data for analysis.

# Troubleshooting


CUDA Out of Memory: Reduce NUM_AGENTS or GRID_SIZE.
CuPy Errors: Ensure CuPy matches CUDA toolkit and GPU drivers.
Slow Performance: Verify GPU execution and kernel fusion.

# License


Provided as-is for educational and research purposes.AcknowledgmentsBuilt with CuPy for GPU acceleration.
Inspired by 3D multi-agent systems and grid-based simulations.

