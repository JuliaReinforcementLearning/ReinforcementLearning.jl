@def title = "General Pipeline for Offline Reinforcement Learning Evaluation Report"
@def description = """
    This is a technical report of the Summer OSPP project [Establish a General Pipeline for Offline Reinforcement Learning Evaluation](https://summer.iscas.ac.cn/#/org/prodetail/210370741?lang=en) used for mid-term evaluation. It provides an overview into the project and also takes a deep dive into the usage and implementation. It also discusses the implications and future scope of this project.
    """
@def is_enable_toc = true
@def has_code = true
@def has_math = true

@def front_matter = """
    {
        "authors": [
            {
                "author":"Prasidh Srikumar",
                "authorURL":"https://github.com/Mobius1D",
                "affiliation":"National Institute of Technology - Trichy",
                "affiliationURL":"https://www.nitt.edu/"
            }
        ],
        "publishedDate":"2021-08-15",
        "citationText":"Prasidh Srikumar, 2021"
    }"""

@def bibliography = "bibliography.bib"

## 1. Introduction

### Project Name

Establish a General Pipeline for Offline Reinforcement Learning Evaluation

### Background

In recent years, there have been several breakthroughs in the field of Reinforcement Learning with numerous practical applications where RL bots have been able to achieve superhuman performance. This is also reflected in the industry where several cutting edge solutions have been developed based on RL ([Tesla Motors](https://www.tesla.com/), [AutoML](https://cloud.google.com/automl), [DeepMind data center cooling solutions](https://deepmind.com/blog/article/deepmind-ai-reduces-google-data-centre-cooling-bill-40) just to name a few).

One of the most prominent challenges in RL is the lack of reliable environments for training RL agents. **Offline RL** has played a pivotal role in solving this problem by removing the need for the agent to interact with the environment to improve its policy over time. This brings forth the problem of not having reliable tests to verify the performance of RL algorithms. Such tests are facilitated by standard datasets ([RL Unplugged](https://arxiv.org/abs/2006.13888)\dcite{gulcehre2020rl}, [D4RL](https://arxiv.org/abs/2004.07219)\dcite{fu2020d4rl} and [An Optimistic Perspective on Offline Reinforcement Learning](https://arxiv.org/abs/1907.04543)\dcite{agarwal2020optimistic}) that are used to train Offline RL agents and benchmark against other algorithms and implementations. [ReinforcementLearningDatasets.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/tree/master/src/ReinforcementLearningDatasets) provides a simple solution to access various standard datasets that are available for Offline RL benchmarking across a variety of tasks.

Another problem in Offline RL is Offline Model Selection. For this, there are numerous policies that are available in [Benchmarks for Deep Off-Policy Evaluation](https://openreview.net/forum?id=kWSeGEeHvF8)\dcite{fu2021benchmarks}. ReinforcementLearningDatasets.jl will also help in loading policies that will aid in model selection in ReinforcementLearning.jl package.

## 2. Project Overview

### Objectives

Create a package called **ReinforcementLearningDatasets.jl** that would aid in loading various standard datasets and policies that are available. Currently supported datasets are:

- [D4RL: Datasets for Deep Data-Driven Reinforcement Learning](https://github.com/rail-berkeley/d4rl)
- [An Optimistic Perspective on Offline Reinforcement Learning (ICML, 2020)](https://github.com/google-research/batch_rl)
- [d4rl-pybullet](https://github.com/takuseno/d4rl-pybullet)
- [RL Unplugged: Benchmarks for Offline Reinforcement Learning](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged)

Make standard policies in [Benchmarks for Deep Off-Policy Evaluation](https://github.com/google-research/deep_ope) to be available in RLDatasets.jl.

Implement an **Off Policy Evaluation** method and select between a number of standard policies for a particular task using RLDatasets.jl.

The following are the future work that are possible in this project.

- Parallel loading and partial loading of datasets.
- Add support for `environments` that are not supported by GymEnvs -> Flow and CARLA.
- Add support for datasets in Flow and CARLA envs.
- Add support for creating, storing and loading custom made datasets.
- `test-train` split functionality for datasets.
- Cross validation and grid search.
- Enable features that make a particular algorithm based on the requirements of the env.
- `evaluator` function that performs evaluation (can be on policy or off policy)
- Metrics as hooks. Refer [Metrics](https://d3rlpy.readthedocs.io/en/v0.90/references/metrics.html)

Refer the following [discussion](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/discussions/359) for more ideas.

### Time Planning

| Date       | Goals |
| :-----------: | :---------: |
| 07/01 - 07/14 | Brainstorm various ideas that are possible for the implementation of RLDatasets.jl and finalize the key features. |
| 07/15 - 07/20 | Made a basic julia wrapper for `d4rl` environments and add some tests |
| 07/21 - 07/30 | Implemented `d4rl` and `d4rl-pybullet` datasets |
| 07/31 - 08/06 | Implemented  `Google Research DQN Replay Datasets` |
| 08/07 - 08/14 | Implemented `RL Unplugged atari datasets`, setup the docs, added README.md. Made the package more user friendly. Make the **mid-term report** |
| 08/15 - 08/30 | Add lazy loading and multi threaded loading support for `Google Research DQN Replay Datasets`. Add the rest of RL Unplugged datasets, polish the interface, finalize the structure of the codebase. Add examples and `register the package`.|
| 09/01 - 09/15 | Add support for policy loading from [Benchmarks for Deep Off-Policy Evaluation](https://github.com/google-research/deep_ope) and implement an `Off Policy Evaluation` method |
| 09/16 - 09/30 | Test OPE in various datasets and publish benchmarks in RLDatasets.jl. Implement other features that makes the package more user friendly. Complete the **final-term report** |

## 3. Implemented datasets

### D4RL

Added support for D4RL datasets with all features loaded in the returned type.

Credits: [D4RL](https://github.com/rail-berkeley/d4rl)

```julia
using ReinforcementLearningDatasets
ds = dataset(
        "hopper-medium-replay-v0";
        repo="d4rl")
```

The type (`D4RLDataSet`) returned by `dataset` is an `Iterator` that returns batches of data based on the requirement that is specified.

Now, you could `take` the values of the `ds` or `iterate` over it.

```
julia> batches = Iterators.take(ds, 2)
D4RLDataSet{StableRNGs.LehmerRNG}(Dict{Symbol, Any}(:reward => Float32[0.9236555, 0.8713692, 0.92237693, 0.9839225, 0.91540813, 0.8331875, 0.8102179, 0.78385466, 0.7304337, 0.6942671  …  5.0350657, 5.005931, 4.998442, 4.986662, 4.9730926, 4.9638906, 4.9503803, 4.9326644, 4.8952913, 4.8448896], :state => Float32[1.2521756 1.2519351 … 0.72994494 0.7145643; 0.00026937472 -0.0048946342 … 0.13946348 0.15210924; … ; 0.002733759 -1.1853988 … -0.06101464 -0.045892276; -0.0028058232 0.08466121 … -1.4235892 -1.0558393], :action => Float32[-0.67060924 -0.39061046 … -0.15234122 -0.1382414; -0.9329903 0.65977097 … 0.9518685 0.9666188; 0.010210991 -0.073685646 … 0.24721281 -0.2440847], :terminal => Int8[0, 0, 0, 0, 0, 0, 0, 0, 0, 0  …  0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), "d4rl", 200919, 256, (:state, :action, :reward, :terminal, :next_state), StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7), Dict{String, Any}("timeouts" => Int8[0, 0, 0, 0, 0, 0, 0, 0, 0, 0  …  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), true)

julia> typeof(batches)
Base.Iterators.Take{D4RLDataSet{StableRNGs.LehmerRNG}}

julia> batch = collect(batches)[1]
NamedTuple{(:state, :action, :reward, :terminal, :next_state), Tuple{Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Int8}, Matrix{Float32}}}

julia> size(batch[:state])
(11, 256)
```

### d4rl-pybullet

Added support for datasets released in `d4rl-pybullet`. This enables testing the agents in complex environments without `Mujoco` license.

Credits: [d4rl-pybullet](https://github.com/takuseno/d4rl-pybullet)

```julia
using ReinforcementLearningDatasets
ds = dataset(
        "hopper-bullet-mixed-v0";
        repo="d4rl-pybullet",
    )
samples = Iterators.take(ds, 2)

```
The output is similar to D4RL.

### Google Research Atari DQN Replay Datasets

Added support for `Google Research Atari DQN Replay Datasets`. Currently, the datasets are directly loaded into the RAM and therefore, it is advised to be used only with sufficient amount of RAM. Support for lazy parallel loading in a `Channel` will be given soon. 

Credits: [DQN Replay Datasets](https://github.com/google-research/batch_rl)

```julia
using ReinforcementLearningDatasets
ds = dataset(
        "pong",
        1,
        [1, 2]
    )
samples = Iterators.take(ds, 2)
```

The output is similar to D4RL.

### RL Unplugged Atari Dataset

Added support for `RL Unplugged` atari datasets. The datasets that are stored in the form of `.tfrecord` are fetched into julia. Lazy loading with multi threading is implemented. This implementation is based on previous work in `TFRecord.jl`.

Credits: [RL Unplugged](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged)

```julia
using ReinforcementLearningDatasets
ds = ds = rl_unplugged_atari_dataset(
        "Pong",
        1,
        [1, 2]
    )
```

The type that is returned is a `Channel{RLTransition}` which returns batches with the given specifications from the buffer when `take!` is used. The point to be noted here is that it takes seconds to load the datasets into the `Channel` and the loading is highly customizable.

```
julia> ds = ds = rl_unplugged_atari_dataset(
               "Pong",
               1,
               [1, 2]
           )
[ Info: Loading the shards [1, 2] in 1 run of Pong with 4 threads
Progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:08
Channel{ReinforcementLearningDatasets.RLTransition}(12) (12 items available)
```

It also supports lazy downloading of the datasets based on the `shards` that are required by the user. In this case only `gs://rl_unplugged/atari/Pong/atari_Pong_run_1-00001-of-00100` and `gs://rl_unplugged/atari/Pong/atari_Pong_run_1-00002-of-00100` will only be downloaded with permissions from the user. If it is already present the `dataset` is located using `DataDeps.jl`. 

The loading time for batches is also very minimal.

```
julia> @time batch = take!(ds)
0.000011 seconds (1 allocation: 80 bytes)

julia> typeof(batch)
ReinforcementLearningDatasets.RLTransition

julia> typeof(batch.state)
Array{UInt8, 4}

julia> size(batch.state)
(84, 84, 4, 256)

julia> size(batch.reward)
(256,)
```


### Relevant commits, discussions and PRs

- [Updated RLDatasets.jl #403](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/403)
- [Expand to d4rl-pybullet #416](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/416)
- [Add Atari datasets released by Google Research #429](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/429)
- [RL unplugged implementation with tests #452](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/452)
- [Features for Offline Reinforcement Learning Pipeline #359](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/discussions/359)
- [Fix record_type issue #24](https://github.com/JuliaReinforcementLearning/TFRecord.jl/pull/24)

## 4. Implementation Details and Challenges Faced

The challenge that was faced during the first week was to chart out a direction for RLDatasets.jl. I researched the implementations of the pipeline in [d3rlpy](https://github.com/takuseno/d3rlpy), [TF.data.Dataset](https://www.tensorflow.org/datasets) etc and then narrowed down some inspiring ideas in the [discussion](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/discussions/359).

Later, I made the [implementation](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/384) as a wrapper around d4rl python library, which was discarded as it did not align with the purpose of the library of being lightweight and not requiring a `Mujoco license` for usage of open source datasets. A wrapper would also not give the fine grained control that we could get if we load the datasets natively.

We decided to use [DataDeps.jl](https://github.com/oxinabox/DataDeps.jl) for registering, tracking and locating datasets without any hassle. [DataDeps.jl](https://github.com/JuliaComputing/DataSets.jl) is a package that helps make data wrangling code more reusable and was crucial in making RLDatasets.jl seamless.

What I learnt here was how to make a package, manage its dependencies and choose which package would be the right fit for the job. I also learnt about `Iterator` interfaces in julia to convert the type (that is output by the `dataset` function) into an `Iterator`. `d4rl-pybullet` was also implemented in a similar fashion.

Implementation of `Google Research Atari DQN Replay Datasets` was harder because it was quite a large dataset and even one shard didn't fit into memory of my machine. I also had to figure out how the data was stored and how to retrieve it. Initially, I planned to use `GZip.jl` to unpack the gzip files and use `NPZ.jl` to read the files. Since, NPZ didn't support reading from `GZipStream` by itself, I had to adapt the functions in `NPZ` to read the stream. Later, we decided to use `CodecZlib` to get a decompressed buffer channel output which was natively supported by `NPZ`. We also had to test it internally and skip the CI test because CI wouldn't be able to handle the dataset. Exploring the possibility of lazy loading of the files that are available and enabling it is also within the scope of the project.

For supporting `RL Unplugged dataset` I had to learn about `.tfrecord` files, `Protocol Buffers`, `buffered Channels` and julia `multi threading` which was used in a lot of occasions. It took some time to grasp all the concepts but the final implementation, however, was based on already existing work in `TFRecord.jl`.

All of this work wouldn't have been possible without the patient mentoring and vast knowledge of my mentor [Jun Tian](https://github.com/findmyway), who has been pivotal in the design and implementation of the package. His massive experience and beautifully written code has provided a lot of inspiration to the making of this package. His amicable nature and commitment to the users of the package by providing timely and detailed explanations to any issues or queries related to the package despite his time constraints, has provided a long standing example as a developer and as a person. I also thank all the developers of the packages that `RLDatasets.jl` depend upon.

### Implementation details

#### Directory Structure

The `src` directory hosts the working logic of the package.

```
src
├─ ReinforcementLearningDatasets.jl
├─ atari
│  ├─ atari_dataset.jl
│  └─ register.jl
├─ common.jl
├─ d4rl
│  ├─ d4rl
│  │  └─ register.jl
│  ├─ d4rl_dataset.jl
│  └─ d4rl_pybullet
│     └─ register.jl
├─ init.jl
└─ rl_unplugged
   ├─ atari
   │  ├─ register.jl
   │  └─ rl_unplugged_atari.jl
   └─ util.jl
```

The directory for handling each dataset would consist of two files. The `register.jl` that would register the `DataDeps` that are required and another file that is responsible for loading the datasets. The `init` functions are called in the project `__init__` for registering right after it is imported.

```julia
function __init__()
    RLDatasets.d4rl_init()
    RLDatasets.d4rl_pybullet_init()
    RLDatasets.atari_init()
    RLDatasets.rl_unplugged_atari_init()
end
```

#### D4RL Datasets implementation

The `register.jl` for d4rl dataset is located in `src/d4rl/d4rl` which registers the `DataDeps`. The following is an example code for the registration.

```julia
function d4rl_init()
    repo = "d4rl"
    for ds in keys(D4RL_DATASET_URLS)
        register(
            DataDep(
                repo*"-"* ds,
                """
                Credits: https://arxiv.org/abs/2004.07219
                The following dataset is fetched from the d4rl. 
                The dataset is fetched and modified in a form that is useful for RL.jl package.
                
                Dataset information: 
                Name: $(ds)
                $(if ds in keys(D4RL_REF_MAX_SCORE) "MAXIMUM_SCORE: " * string(D4RL_REF_MAX_SCORE[ds]) end)
                $(if ds in keys(D4RL_REF_MIN_SCORE) "MINIMUM_SCORE: " * string(D4RL_REF_MIN_SCORE[ds]) end) 
                """, #check if the MAX and MIN score part is even necessary and make the log file prettier
                D4RL_DATASET_URLS[ds],
            )
        )
    end
    nothing
end
```

The dataset is loaded using `ReinforcementLearningDatasets/src/d4rl/d4rl_dataset.jl` and is enclosed in a `D4RLDataSet` type.

```julia
struct D4RLDataSet{T<:AbstractRNG} <: RLDataSet
    dataset::Dict{Symbol, Any}
    repo::String
    dataset_size::Integer
    batch_size::Integer
    style::Tuple
    rng::T
    meta::Dict
    is_shuffle::Bool
end
```

The dataset function is used to retrieve the files.

```julia
function dataset(dataset::String;
    style=SARTS,
    repo = "d4rl",
    rng = StableRNG(123), 
    is_shuffle = true, 
    batch_size=256
)
```

The dataset is downloaded if the dataset is not present and loaded from the local file system using `DataDeps.jl` 

```julia
try 
    @datadep_str repo*"-"*dataset 
catch 
    throw("The provided dataset is not available") 
end
    
path = @datadep_str repo*"-"*dataset 

@assert length(readdir(path)) == 1
file_name = readdir(path)[1]

data = h5open(path*"/"*file_name, "r") do file
    read(file)
end
```

The dataset is loaded into `D4RLDataSet` `Iterator` and returned. The iteration logic is also implemented in the same file using `Iterator` interfaces.

#### RL Unplugged Atari

Some of the interesting pieces of code used in loading RL Unplugged dataset.

Multi threaded iteration over a `Channel{Example}` to `put!` into another `Channel{RLTransition}`.

```julia
ch_src = Channel{RLTransition}(n * tf_reader_sz) do ch
    for fs in partition(shuffled_files, n)
        Threads.foreach(
            TFRecord.read(
                fs;
                compression=:gzip,
                bufsize=tf_reader_bufsize,
                channel_size=tf_reader_sz,
            );
            schedule=Threads.StaticSchedule()
        ) do x
            put!(ch, RLTransition(x))
        end
    end
end
```

Multi threaded batching using a parallel loop where each thread loads the batches into `Channel{RLTransition}`.

```julia
res = Channel{RLTransition}(n_preallocations; taskref=taskref, spawn=true) do ch
    Threads.@threads for i in 1:batch_size
        put!(ch, deepcopy(batch(buffer_template, popfirst!(transitions), i)))
    end
end
```

## 5. Implications

Equipping RL.jl with RLDatasets.jl is a key step in making the package more industry relevant because different offline algorithms can be compared with respect to a variety of standard offline dataset benchmarks. It is also meant to improve the implementations of existing offline algorithms and make it on par with the SOTA implementations. This package provides a seamless way of downloading and accessing existing datasets and also supports loading datasets into memory with ease, which if implemented separately, would be tedious for the user.

After the implementation of [Benchmarks for Deep Off-Policy Evaluation](https://github.com/google-research/deep_ope), testing and comparing algorithms would be much easier than before. This package would also make SOTA offline RL more accessible and reliable than ever before in ReinforcementLearning.jl.


## 6. Future Plan

### Within the time frame of the project.

Within the scope of the project and in the given time frame, we are planning to:

- Polish the package in terms of structure and make it more user friendly. 
- Support the datasets that have not been added.
- Support [Benchmarks for Deep Off-Policy Evaluation](https://github.com/google-research/deep_ope) using`ONNX.jl`. Support possible policies that are provided for implementation within the time frame.
- Write experiments for policy selection using RLDatasets.jl to finalize and establish the usability of the package.

### Further down the line 

Enabling more features mentioned in [Features for Offline Reinforcement Learning Pipeline #359](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/discussions/359) would be the next obvious step after implementing an OPE method (like FQE). Dataset generation, storage and policy parameter storage would also be great to implement in RLDatasets.jl.