# Multi-Agent Grid World Simulation with CUDA

This project implements a simple multi-agent grid world environment where 1,000 agents move randomly on a 2D grid, with movement and collision detection accelerated using CUDA via the CuPy library. The simulation runs on a GPU for high performance, making it suitable for large-scale agent-based modeling.

# Overview

Environment: A 100x100 2D grid where agents occupy discrete cells.
Agents: 1,000 agents, each moving randomly (up, down, left, right) to an adjacent cell if it is empty.
Collision Handling: Agents cannot move to occupied cells and remain in place if a collision is detected.
Acceleration: Uses CuPy to perform agent updates and collision checks in parallel on a CUDA-enabled GPU.
Simulation: Runs for a configurable number of steps (default: 100).

# Features

Random initialization of agent positions.
GPU-accelerated movement and collision detection using CuPy.
Optimized CUDA kernel execution with kernel fusion.
Configurable grid size, number of agents, and simulation steps.
# Requirements

Hardware: A CUDA-enabled GPU with sufficient memory (e.g., NVIDIA GPU with CUDA support).
Software:Python 3.8+
CuPy (CUDA-compatible NumPy-like library)
NumPy

CUDA Toolkit: Ensure your GPU driver and CUDA version are compatible with CuPy.


# Installation
Install Python dependencies:bash
pip install cupy-cuda12x numpy

CUDA version for your system (e.g., cuda11x). Check CuPy's documentation for details.
Verify your GPU is CUDA-compatible and drivers are installed:Run nvidia-smi in your terminal to check GPU status.
Ensure the CUDA toolkit is installed and matches your CuPy version.

# Usage
Save the main script as multi_agent_grid_world.py.
Run the simulation:
"bash
python multi_agent_grid_world.py
The script will:Initialize 1,000 agents on a 100x100 grid.
Run the simulation for 100 steps.
Print progress every 10 steps and the total runtime.
Output the final positions of the first 5 agents.

# Configuration

Modify the following constants in multi_agent_grid_world.py to customize the simulation:GRID_SIZE: Size of the square grid (default: 100).
NUM_AGENTS: Number of agents (default: 1,000).
NUM_STEPS: Number of simulation steps (default: 100).
SEED: Random seed for reproducibility (default: 42).


Example Output
Initialized 1000 agents on a 100x100 grid.
Step 10/100 completed.
Step 20/100 completed.
...
Step 100/100 completed.
Simulation completed in 0.15 seconds.
Final positions of first 5 agents:
[[45 67]
 [23 89]
 [12 34]
 [78 56]
 [90 22]]

 # Implementation Details
 
 Initialization: Agents are assigned random (x, y) positions using CuPy's random number generator.
Movement: Each agent selects a random direction (up, down, left, right). Positions are updated in parallel, with boundary checks to keep agents within the grid.
Collision Detection: A grid tracks occupied cells. Moves to occupied cells are rejected, and agents stay in place.
CUDA Acceleration: CuPy handles array operations on the GPU. The @cp.fusion.fuse() decorator optimizes kernel performance.
Error Handling: Catches CUDA out-of-memory errors and suggests reducing NUM_AGENTS or GRID_SIZE.

# Limitations

Collision Detection: Uses a simple grid-based approach, which may be slow for very dense grids. Consider spatial partitioning for better scalability.
Visualization: Not included to keep the code minimal. Add Matplotlib or similar for visual output (requires transferring data to CPU with cp.asnumpy()).
Agent Behavior: Limited to random movement. Extend the update_positions function for more complex behaviors (e.g., goal-directed movement).

# Potential Extensions

Add visualization using Matplotlib or a real-time graphics library.
Implement advanced agent behaviors (e.g., pathfinding, interaction rules).
Optimize collision detection with spatial data structures (e.g., quadtrees).
Add logging or data export for analysis (e.g., agent trajectories).

# Troubleshooting


CUDA Out of Memory: Reduce NUM_AGENTS or GRID_SIZE, or use a GPU with more memory.
CuPy Import Errors: Ensure the CuPy version matches your CUDA toolkit and GPU drivers. Check the CuPy installation guide.
Slow Performance: Verify that the simulation is running on the GPU (CuPy operations) and not falling back to CPU.


# License

This project is provided as-is for educational and research purposes. No warranty is implied.




# AcknowledgmentsBuilt with CuPy for GPU acceleration.
Inspired by multi-agent simulation frameworks and grid-based environments.







