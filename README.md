# COSC 4368 Maze Solver - Checkpoint 2

## Overview
This checkpoint demonstrates:
- loading the maze in Python,
- solving the maze without hazards using BFS,
- visualizing the solution path,
- loading hazards,
- demonstrating hazard behavior.

## Files
- `maze.py` - converts the maze image into `maze_walls.npy`
- `maze_config.py` - stores start, goal, and hazard positions
- `hazards.py` - validates hazard coordinates
- `environment.py` - environment logic and hazard mechanics
- `solver.py` - BFS pathfinding
- `visualizer.py` - maze visualization
- `demo_checkin2.py` - checkpoint demo runner

## How to run

1. Build the wall matrix:
```bash
python maze.py