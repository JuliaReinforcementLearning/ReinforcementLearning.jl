@def title = "Establish a General Pipeline for Offline Reinforcement Learning Evaluation (Summer OSPP Project 210370741) Mid-term Report"
@def description = """
    This is a technical report of the summer OSPP project [Establish a General Pipeline for Offline Reinforcement Learning Evaluation](https://summer.iscas.ac.cn/#/org/prodetail/210370741?lang=en) used for mid-term evaluation. The report is split into the following parts:
    - `Introduction`
    - `Project Overview`
    - `Implemented Datasets`
    - `Implementation Details and Challenges Faced`
    - `Implications`
    - `Future Plan`
    """
@def is_enable_toc = true
@def has_code = true
@def has_math = true

@def front_matter = """
    {
        "authors": [
            "author":"Prasidh Srikumar",
            "authorURL":"https://github.com/Mobius1D"
            "affiliation":"",
            "affiliationURL":""
        ]
    },
    "publishedDate":"2021-08-14",
    "citationText":"Prasidh Srikumar, 2021"
    }"""

@def bibliography = "bibliography.bib"

## 1. Introduction

### Project Name

Establish a General Pipeline for Offline Reinforcement Learning Evaluation

### Background

In the recent years, there has been several breakthroughs in the field of Reinforcement Learning with numerous practical applications where RL bots have been able to achieve superhuman performance. This is also reflected in the industry where several cutting edge solutions have been developed based on RL (Tesla Motors, AutoML, DeepMind data center cooling solutions just to name a few).

One of the most notorious challenges in RL is the lack of reliable environments for training RL agents. Offline RL has played a pivotal role in solving this problem by removing the need for the agent to interact with the environment to improve its policy over time. This brings forth the problem of having reliable tests to verify the performance of RL algorithms. Such tests are facilitated by standard datasets ([RL Unplugged](https://arxiv.org/abs/2006.13888), [D4RL](https://arxiv.org/abs/2004.07219) to name a few) that are used to train Offline RL agents and benchmark against other algorithms and implementations. [ReinforcementLearningDatasets.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/tree/master/src/ReinforcementLearningDatasets) provides a simple solution to access various standard datasets that are available for Offline RL benchmarking across a variety of tasks.

Another problem in Offline RL is Offline Model Selection. For this there are several policies that are available in [Benchmarks for Deep Off-Policy Evaluation](https://github.com/google-research/deep_ope). ReinforcementLearningDatasets.jl will also help in loading policies that will aid in model selection in ReinforcementLearning.jl package.

## 2. Project Overview

### Objectives

Create a package called **ReinforcementLearningDatasets.jl** that would aid in loading various standard datasets and policies that are available.

Make the following datasets available in RLDatasets.

- [D4RL: Datasets for Deep Data-Driven Reinforcement Learning](https://github.com/rail-berkeley/d4rl)
- [An Optimistic Perspective on Offline Reinforcement Learning (ICML, 2020)](https://github.com/google-research/batch_rl)
- [d4rl-pybullet](https://github.com/takuseno/d4rl-pybullet)'
- [RL Unplugged: Benchmarks for Offline Reinforcement Learning](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged)

Make standard policies in [Benchmarks for Deep Off-Policy Evaluation](https://github.com/google-research/deep_ope) to be available in RLDatasets.jl.

Implement an OPE method and select between a number of standard policies for a particular task using RLDatasets.jl.

Following are the future work that are possible in this project.

- Parallel loading and partial loading of datasets for supported datasets.
- Add support for envs that are not supported by GymEnvs -> Flow and CARLA.
- Add support for datasets in Flow and CARLA envs.
- Add support for creating, storing and loading custom made datasets.
- test-train split functionality for datasets.
- cross validation and grid search.
- build_with_dataset - make a particular algorithm based on the requirements of the env.
- evaluator function that performs evaluation (can be on policy or off policy)
- Metrics as hooks. Refer [Metrics](https://d3rlpy.readthedocs.io/en/v0.90/references/metrics.html)

Refer the following [discussion](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/discussions/359) for more ideas that could be possible.

### Time Planning

| Date       | Goals |
| :-----------: | :---------: |
| 07/01 - 07/14 | Brainstorm various ideas that are possible for the implementation of RLDatasets.jl and finalize the key features. |
| 07/15 - 07/20 | Made a basic julia wrapper for `d4rl` environments and add some tests |
| 07/21 - 07/30 | Implement loading of `d4rl` and `d4rl-pybullet` datasets |
| 07/31 - 08/06 | Implement loading of `Google Research DQN Replay Datasets` |
| 08/07 - 08/14 | Implement loading of `RL Unplugged atari datasets`, setup the docs, add README.md. Make the package more user friendly. Make the **mid-term report** |
| 08/15 - 08/30 | Add lazy loading and multi threaded loading support for `Google Research DQN Replay Datasets`. Add the rest of RL Unplugged datasets, polish the interface, finalize the structure of the codebase. Add examples and `register the package.` |
| 09/01 - 09/15 | Add support for policy loading from [Benchmarks for Deep Off-Policy Evaluation](https://github.com/google-research/deep_ope) and implement an OPE method |
| 09/16 - 09/30 | Test OPE in various datasets and publish benchmarks in RLDatasets.jl. Implement other features that makes the package more user friendly. Complete the **final-term report** |

## 3. Implemented datasets

### D4RL

Added support for D4RL datasets with all features loaded in the returned type.

Credits: https://github.com/rail-berkeley/d4rl

```julia
using ReinforcementLearningDatasets
ds = dataset(
        "hopper-medium-replay-v0";
        repo="d4rl")
samples = take!(ds, 2)
```

The type (`D4RLDataSet`) returned by `dataset` is an iterable that returns batches of data based on the requirement that is specified.

### d4rl-pybullet

Added support for datasets released in `d4rl-pybullet`. This enables testing the agents in complex environments without `Mujoco` license.

Credits: https://github.com/takuseno/d4rl-pybullet

```julia
using ReinforcementLearningDatasets
ds = dataset(
        "hopper-bullet-mixed-v0";
        repo="d4rl-pybullet",
    )
samples = take!(ds, 2)
```
The output is similar to D4RL.

### Google Research Atari DQN Replay Datasets

Added support for `Google Research Atari DQN Replay Datasets`. Currently, the datasets are directly loaded into the RAM and therefore it is advised to be used only with sufficient amount of RAM. Support for lazy parallel loading in a `Channel` will be given soon. 

Credits: https://github.com/google-research/batch_rl

```julia
using ReinforcementLearningDatasets
ds = dataset(
        "pong",
        1,
        [1, 2]
    )
samples = take!(ds, 2)
```

The output is similar to D4RL.

### RL Unplugged atari dataset

Added support for `RL Unplugged` atari datasets. The datasets that are stored in the form of `.tfrecord` are fetched into julia. Lazy loading with multi threading is implemented. This implementation is based on previous work in `TFRecord.jl`.

Credits: https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged

```julia
using ReinforcementLearningDatasets
ds = ds = rl_unplugged_atari_dataset(
        "Pong",
        1,
        [1, 2]
    )
samples = take!(ds)
```

The type that is returned is a `Channel{RLTransition}` which returns batches of the given specifications from the buffer when `take!` is used.

### Relevant commits, discussions and PRs

- [Updated RLDatasets.jl #403](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/403)
- [Expand to d4rl-pybullet #416](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/416)
- [Add Atari datasets released by Google Research #429](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/429)
- [RL unplugged implementation with tests #452](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/452)
- [Features for Offline Reinforcement Learning Pipeline #359](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/discussions/359)
- [Fix record_type issue #24](https://github.com/JuliaReinforcementLearning/TFRecord.jl/pull/24)

## 4. Implementation Details and Challenges Faced

The challenge that was faced during the first week was to chart out a direction for RLDatasets.jl. So, I had to research implementations of the pipeline in [d3rlpy](https://github.com/takuseno/d3rlpy), [TF.data.Dataset](https://www.tensorflow.org/datasets) etc and then narrowed down some of the inspiring ideas in the [discussion](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/discussions/359).

Later I made the [implementation](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/384) as a wrapper around d4rl python library which was discarded as it did not align with the purpose of the library of being lightweight and not requiring a `Mujoco license` for usage of open source datasets. A wrapper would also not give the fine grained control that we could get if we load the datasets natively.

We decided to use [DataDeps.jl](https://github.com/oxinabox/DataDeps.jl) for registering, tracking and locating datasets without any hassle. What I learnt here was how to make a package, manage its dependencies and choose which package would be the right fit for the job. I also learnt about `Iterator` interfaces in julia to convert the type (that is output by the `dataset` function) into an `iterator`. `d4rl-pybullet` was also implemented in a similar fashion.

Implementation of `Google Research Atari DQN Replay Datasets` was harder because it was quite a large dataset and even one shard didn't fit into memory of my machine. One of the major things that I had to figure out was how the data was stored and how to retrieve it. Initially I planned to use `GZip.jl` to unpack the gzip files and use `NPZ.jl` to read the files. Since, NPZ didn't support reading from `GZipStream` by itself, I had to adapt the functions in `NPZ` to read the stream. Later we decided to use `CodecZlib` to get a decompressed buffer channel output which was natively supported by `NPZ`. We also had to test it internally and skip the CI test because CI wouldn't be able to handle the dataset. Exploring the possibility of lazy loading of the files that are available and enabling it is also within the scope of the project.

For supporting `RL Unplugged dataset` I had to learn about `.tfrecord` files, `Protocol Buffers`, `buffered Channels` and julia `multi threading` which was used in a lot of occasions. It took some time to grasp all the concepts but the final implementation was however based on already existing work in `TFRecord.jl`.

All of this work wouldn't have been possible without the patient mentoring and vast knowledge that was shown by my mentor [Jun Tian](https://github.com/findmyway) who has been pivotal in the design and implementation of the package. His massive experience and beautifully written code has provided a lot of inspiration for the making of this package. His amicable nature and commitment to the users of the package providing timely and detailed explanations to any issues or queries related to the package despite his time constraints has provided a long standing example as a developer and as a person for the developers within and outside OSPP.

### Implementation details

#### Directory Structure

The `src` directory hosts the working logic of the package.

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

The dataset is loaded using `ReinforcementLearningDatasets/src/d4rl/d4rl_dataset.jl` and enclosed in a `D4RLDataSet` type.

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

The dataset is loaded into `D4RLDataSet` iterable and returned. The iteration logic is also implemented in the same file using `Iterator` interfaces.

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

Equipping RL.jl with RLDatasets.jl is a key step in making the package more industry relevant because different offline algorithms can be compared with respect to a variety of standard offline dataset benchmarks. It is also meant to improve the implementations of existing offline algorithms and make it on par with the SOTA implementations. This package provides a seamless way of downloading and accessing existing datasets and also supports loading datasets into memory with ease which implemented separately would be tedious for the user.

After the implementation of [Benchmarks for Deep Off-Policy Evaluation](https://github.com/google-research/deep_ope), testing and comparing algorithms would be much easier than before. This package would also help SOTA offline RL more accessible and reliable than ever before in ReinforcementLearning.jl.


## 6. Future Plan

### Within the time frame of the project.

Within the scope of the project and in the given time frame we are planning to:

- Polish the package in terms of structure and make it more user friendly. 
- Support the datasets that has not been added.
- Support of [Benchmarks for Deep Off-Policy Evaluation](https://github.com/google-research/deep_ope) using`ONNX.jl`. Support possible policies that are provided for implementation within the time frame.
- Experiments for policy selection using RLDatasets.jl to finalize and establish the usability of the package.

### Ideas further down the line 

Enabling more features that were mentioned in [Features for Offline Reinforcement Learning Pipeline #359](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/discussions/359) would be the next obvious step after the implementation an OPE method (like FQE) has been explored. Dataset generation, storage and policy parameter storage would also be great to implement in RLDatasets.jl.