# AI-vaccum-agent

## Overview
The project is centered around designing and implementing common search algorithms for intelligent agents in typical environments. The environment consists of a 2D grid of rooms, some of which are 'dirty' and some might be blocked. The goal is to use these search algorithms to help the vacuuming robot navigate and clean the rooms efficiently.

The starting point is the center of the grid, and the objective is to find the 'closest' dirty room based on the chosen search algorithm. The agent should then find the path to that room and repeat the process for the next dirty room until all rooms are clean.

## Implemented Algorithms
- Breadth-First Graph Search (BF Graph)

- Depth-First Graph Search (DF Graph)

- Uniform Cost Search (UC)

- A* Search (A*)

## Technology
- **PyGUI**: The project's graphical user interface (GUI) was designed and implemented using PygUI, a Python library for creating interactive graphical interfaces. PygUI enabled us to visually showcase the search algorithms' execution and the agent's movement across the grid.