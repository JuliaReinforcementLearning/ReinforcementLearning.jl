# ReinforcementLearningDatasets

A package to create, manage, store and retrieve datasets for Offline Reinforcement Learning. This package uses [DataDeps.jl](https://github.com/oxinabox/DataDeps.jl) to fetch and track datasets. For more details refer to the [documentation](https://juliareinforcementlearning.org/docs/rldatasets/).

It supports an extensive number of datasets and also supports [google-research/deep_ope](https://github.com/google-research/deep_ope) d4rl policies.
## Install
```
pkg> add ReinforcementLearningDatasets
```
## Examples

### D4RL dataset
```julia-repl
julia> using ReinforcementLearningDatasets

julia> ds = dataset("hopper-medium-replay-v0"; repo="d4rl")
D4RLDataSet{Random.MersenneTwister}(Dict{Symbol, Any}(:reward => Float32[0.9236555, 0.8713692, 0.92237693, 0.9839225, 0.91540813, 0.8331875, 0.8102179, 0.78385466, 0.7304337, 0.6942671  …  5.0350657, 5.005931, 4.998442, 4.986662, 4.9730926, 4.9638906, 4.9503803, 4.9326644, 4.8952913, 4.8448896], :state => Float32[1.2521756 1.
...

julia> samples = Iterators.take(ds)
Base.Iterators.Take{D4RLDataSet{Random.MersenneTwister}}(D4RLDataSet{Random.MersenneTwister}(Dict{Symbol, Any}(:reward => Float32[0.9236555, 0.8713692, 0.92237693, 0.9839225, 0.91540813, 0.8331875, 0.8102179, 0.78385466, 0.7304337, 0.6942671  …  5.0350657, 5.005931, 4.998442, 4.986662, 4.9730926, 4.9638906, 4.9503803, 4.9326644, 4.8952913, 4.8448896], :state => Float32[1.2521756 1.2519351 … 
...
```
`ds` is of the type `D4RLDataset` which consists of the entire dataset along with some other information about the dataset. `samples` are in the form of `SARTS` with batch_size 256.

### RL Unplugged
```julia-repl
julia> using ReinforcementLearningDatasets

julia> ds = rl_unplugged_atari_dataset("pong", 1, [1, 2])
[ Info: Loading the shards [1, 2] in 1 run of pong with 1 threads
Progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:13
RingBuffer{ReinforcementLearningDatasets.AtariRLTransition}(Channel{ReinforcementLearningDatasets.AtariRLTransition}(12), ReinforcementLearningDatasets.AtariRLTransition(UInt8[0x00 0x00 … 0x00 0x00; 0x00 0x00 … 0x00 0x00; … ; 0x00 0x00 … 0x00 0x00; 0x00 0x00 … 0x00 0x00]
...

julia> samples = take!(ds, 2)
ReinforcementLearningDatasets.AtariRLTransition(UInt8[0x34 0x34 … 0x57 0x57; 0x57 0x57 … 0x57 0x57; … ; 0xec 0xec … 0xec 0xec; 0xec 0xec … 0xec 0xec]

UInt8[0x34 0x34 … 0x57 0x57; 0x57 0x57 … 0x57 0x57; … ; 0xec 0xec … 0xec 0xec; 0xec 0xec … 0xec 0xec]
...
499684941823, -2724510648791728127, 4553719765411037185, -3513317882744274943, -8544304859447295999, -1756940416348848127, 186459579884765185, -9154762511281553407, -1410303982529675263, -5170686526081728511], Float32[18.0, 17.0, 19.0, 18.0, 16.0, 18.0, 12.0, 19.0, 21.0, 21.0  …  20.0, 18.0, 18.0, 21.0, -2.0, -18.0, 14.0, 9.0, -21.0, -15.0])
```
`ds` is a `Channel{AtariRLTransition}` that returns batches of type `AtariRLTransition` when `take!` is used.

### Deep OPE
```julia-repl
julia> using ReinforcementLearningDatasets
julia> model = d4rl_policy("ant", "online", 10)
D4RLGaussianNetwork{Flux.Chain{Tuple{Flux.Dense{typeof(NNlib.relu), Matrix{Float32}, Vector{Float32}}, Flux.Dense{typeof(NNlib.relu), Matrix{Float32}, Vector{Float32}}}}, Flux.Chain{Tuple{Flux.Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}}}, Flux.Chain{Tuple{Flux.Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}}}}(Chain(Dense(111, 256, relu), Dense(256, 256, relu)), Chain(Dense(256, 8)), Chain(Dense(256, 8)))

julia> env = GymEnv("ant-medium-v0")

julia> a = state(env) |> model 
([0.4033523672563252, 0.2595324728828865, -0.5570708932001185, -0.5522664630767464, -0.9880830678905399, 0.26941818745211277, 2.1526997615143517, -0.09209516788500087], [0.1891864047269633, -0.08529361693109125, -0.744898545155567, -0.6052428389550205, -0.8887611225758812, 0.37303904310491376, 1.8524731056470352, -0.08358713385474797])

julia> plt = deep_ope_d4rl_evaluate("halfcheetah", "online", 10; num_evaluations=100)
Progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:17
                      halfcheetah-medium-v0 scores              
               +----------------------------------------+       
         12000 |    .       ...  ...         ..         | scores
               |: .::':..:.:'':::'''.::::::''::'::':'::.|       
               |::::' ::::''  '::    ::      ::  : ' '  |       
               |::::  :::      ::    ::      ::         |       
               |::::  ::'      ':    ''      ::         |       
               |::::  ::                     ::         |       
               |::::   :                     ::         |       
   score       | ::    :                     ::         |       
               | ::    :                     ::         |       
               | ::    :                     ::         |       
               | ::    :                     ::         |       
               | ::    '                     ::         |       
               | ::                          ::         |       
               | ::                          ::         |       
          1000 |                                        |       
               +----------------------------------------+       
                0                                    100        
                                 episode    
```

`d4rl_policy` returns a model that yields a `Tuple` containing `a`(actions) and `μ`(the mean).

## Supported Datasets
* [D4RL: Datasets for Deep Data-Driven Reinforcement Learning](https://github.com/rail-berkeley/d4rl)
    * Mujoco datases and Pybullet datasets provided by D4RL are actively supported.
    * Flow and CARLA datasets have not been tested yet.
    * Mujoco License is not needed to access these datasets but will be required for online evaluation.
* [d4rl-pybullet](https://github.com/takuseno/d4rl-pybullet)
* [Google Research atari DQN replay datasets](https://github.com/google-research/batch_rl)
    * This directly loads the entire dataset into the RAM and should be used with caution as it takes up more than 20 GB of RAM for even a single epoch.
* [RL Unplugged: Benchmarks for Offline Reinforcement Learning](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged)
    * Currently supports all datasets except rwrl in RL Unplugged.
    * Multi threaded data loading is supported using a `Channel` that returns batches when `Base.take!` is used.

## Supported Models for OPE
* [google-research/deep_ope](https://github.com/google-research/deep_ope)
    * D4RL policies are supported as of now.
    * Support for RLUnplugged policies will be given soon.

**Note**: The package is under active development and support for a few datasets are left. Support for GymEnv for the datasets will also be given soon.