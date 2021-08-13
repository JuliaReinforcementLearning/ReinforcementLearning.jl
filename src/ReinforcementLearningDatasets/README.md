# ReinforcementLearningDatasets

A package to create, manage, store and retrieve datasets for Offline Reinforcement Learning using ReinforcementLearning.jl package.

- This package uses DataDeps.jl to fetch and manage packages.

### Note:

The package is under active development and for now it supports d4rl, d4rl-pybullet and Google Research atari DQN replay datasets.

The package also supports RL Unplugged atari datasets which uses multi threading to load the data efficiently. It is also available in a Channel so that all the data need not fit into the memory.