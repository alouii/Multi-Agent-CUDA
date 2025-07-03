#Multi-Agent Grid World Simulation with CUDA

This project implements a simple multi-agent grid world environment where 1,000 agents move randomly on a 2D grid, with movement and collision detection accelerated using CUDA via the CuPy library. The simulation runs on a GPU for high performance, making it suitable for large-scale agent-based modeling.

#Overview

Environment: A 100x100 2D grid where agents occupy discrete cells.
Agents: 1,000 agents, each moving randomly (up, down, left, right) to an adjacent cell if it is empty.
Collision Handling: Agents cannot move to occupied cells and remain in place if a collision is detected.
Acceleration: Uses CuPy to perform agent updates and collision checks in parallel on a CUDA-enabled GPU.
Simulation: Runs for a configurable number of steps (default: 100).

#Features

Random initialization of agent positions.
GPU-accelerated movement and collision detection using CuPy.
Optimized CUDA kernel execution with kernel fusion.
Configurable grid size, number of agents, and simulation steps.
#Requirements

Hardware: A CUDA-enabled GPU with sufficient memory (e.g., NVIDIA GPU with CUDA support).
Software:Python 3.8+
CuPy (CUDA-compatible NumPy-like library)
NumPy

CUDA Toolkit: Ensure your GPU driver and CUDA version are compatible with CuPy.


#


