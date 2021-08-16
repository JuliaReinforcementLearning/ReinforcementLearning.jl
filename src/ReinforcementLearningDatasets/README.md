# ReinforcementLearningDatasets

A package to create, manage, store and retrieve datasets for Offline Reinforcement Learning. This package uses [DataDeps.jl](https://github.com/oxinabox/DataDeps.jl) to fetch and track datasets.
## Install
Since the package is not registered, you could install the package using the following command.
```julia
] add https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/tree/master/src/ReinforcementLearningDatasets
```
## Usage
### Examples
#### D4RL dataset
```julia
using ReinforcementLearningDatasets
ds = dataset("hopper-medium-replay-v0"; repo="d4rl")
samples = Iterators.take!(ds)
```
`ds` is of the type `D4RLDataset` which consists of the entire dataset along with some other information about the dataset. `samples` are in the form of `SARTS` with batch_size 256.
#### RL Unplugged
```julia
using ReinforcementLearningDatasets
ds = rl_unplugged_atari_dataset("pong", 1, [1, 2])
samples = Iterators.take!(ds, 2)
```
`ds` is a `Channel{RLTransition}` that returns batches of type `RLTransition` when `take!` is used.

For more details refer to the documentation.

**Note**: The package is under active development and for now it supports only a limited number of datasets. 

## Supported Datasets
* [D4RL: Datasets for Deep Data-Driven Reinforcement Learning](https://github.com/rail-berkeley/d4rl)
    * Mujoco datases and Pybullet datasets provided by D4RL are actively supported.
    * Flow and CARLA datasets have not been tested yet.
    * Mujoco Licence is not needed to access these Mujoco datasets but will be required for online evaluation.
* [d4rl-pybullet](https://github.com/takuseno/d4rl-pybullet)
* [Google Research atari DQN replay datasets](https://github.com/google-research/batch_rl)
    * This directly loads the entire dataset into the RAM and should be used with caution. Features for multi threaded lazy loading will be provided soon. 
* [RL Unplugged: Benchmarks for Offline Reinforcement Learning](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged)
    * Currently supports atari datasets that are provided in RL Unplugged.
    * Multi threaded data loading is supported using a `Channel` that returns batches when `Base.take!` is used.