# Master_Thesis_Code
This repository contains the basis code created and used throughout the master thesis: "Learning to improve evolutionary computation for a Warehouse Design and Control Problem".

Files:
  - DDQN_Agent: Contains the implementation of the Dueling DQN agent and is used by the DRL_NSGA_III file.
  - DRL_NSGA_III: The actual implementation of the framework. Upon running it will execute an optimization trajectory using the simulation and the agent.
  - Vanilla_NSGA_II: An original implementation of the Non-dominated Sorting Genetic Algorithm II (NSGA-II). Upon running it will execute an optimization trajectory         using the simulation (without a DRL agent).
  - Vanilla_NSGA_III: An original implementation of the Non-dominated Sorting Genetic Algorithm III (NSGA-III). Upon running it will execute an optimization trajectory       using the simulation (without a DRL agent).
  - Vanilla_SMPSO: An original implementation of the Speed Constrained Particle Swarm Optimization (SMPSO). Upon running it will execute an optimization trajectory using     the simulation (without a DRL agent).

Folder:
  - Simulation_Model: Contains all files which comprise the simulation replication used throughout the research as the Warehouse Design and Control Problem (WDCP).
