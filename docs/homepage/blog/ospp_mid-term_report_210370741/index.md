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

In the recent years there have been several breakthroughs in the field of Reinforcement Learning with several practical applications where RL bots have been able to achieve superhuman performance. This has also been reflected in the industry where several cutting edge solutions have been developed based on RL (Tesla Motors, AutoML, DeepMind data center cooling solutions just to name a few). 

One of the most notorious challenges in RL is the lack of reliable environments for training RL agents. Offline RL has played a pivotal role in solving this problem by removing the need for the agent to interact with the environment to improve its policy over time. This brings forth the problem of having reliable tests to verify the performance of RL algorithms. Such tests are facilitated by standard datasets ([RL Unplugged](https://arxiv.org/abs/2006.13888), [D4RL](https://arxiv.org/abs/2004.07219) to name a few) that are used to train Offline RL agents and benchmark against other algorithms and implementations. [ReinforcementLearningDatasets.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/tree/master/src/ReinforcementLearningDatasets) provides a simple solution to access various standard datasets that are available for Offline RL benchmarking across a variety of tasks.

Another problem in Offline RL is Offline Model Selection. For this there are several policies that are available in [Benchmarks for Deep Off-Policy Evaluation](https://github.com/google-research/deep_ope). ReinforcementLearningDatasets.jl will also help in loading policies that will aid in model selection in ReinforcementLearning.jl package.

## Project Overview

### Objectives

Create a package called **ReinforcementLearningDatasets.jl** that would aid in loading various standard datasets that are available.

Make the following datasets available in RLDatasets.

- [D4RL: Datasets for Deep Data-Driven Reinforcement Learning](https://github.com/rail-berkeley/d4rl)
- [An Optimistic Perspective on Offline Reinforcement Learning (ICML, 2020)](https://github.com/google-research/batch_rl)
- [d4rl-pybullet](https://github.com/takuseno/d4rl-pybullet)'
- [RL Unplugged: Benchmarks for Offline Reinforcement Learning](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged)

Make standard policies available in [Benchmarks for Deep Off-Policy Evaluation](https://github.com/google-research/deep_ope). Try making feasible policies available in RLDatasets.jl.

Implement an OPE method and select between a number of standard policies for a particular task using RLDatasets.jl.

Following are the future work that is possible in this project.
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
| 07/15 - 07/20 | Made a basic julia wrapper for `d4rl` environments and add some tests|
| 07/21 - 07/30 | Implement loading of `d4rl` and `d4rl-pybullet` datasets |
| 07/31 - 08/06 | Implement loading of `Google Research DQN Replay Datasets` |
| 08/07 - 08/14 | Implement loading of `RL Unplugged atari datasets`, setup the docs, add README.md. Make the package more user friendly. Make the **mid-term report** |
| 08/15 - 08/30 | Add the rest of RL Unplugged datasets, polish the interface, finalize the structure of the codebase. Add examples and `register the package.` |
| 09/01 - 09/15 | Add support for policy loading from [Benchmarks for Deep Off-Policy Evaluation](https://github.com/google-research/deep_ope) and implement and OPE method |
| 09/16 - 09/30 | Test OPE in various environments and publish benchmarks in RLDatasets.jl. Implement other features that makes the package more user friendly. Complete the **final-term report** |

## Implemented datasets

### D4RL

Added support for D4RL datasets with all features available in the dataset loaded in the returned type.

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

Added support for d4rl pybullet datasets released in `d4rl-pybullet`. This enables testing the agents in complex environments without `Mujoco` license.

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

Added support for `Google Research Atari DQN Replay Datasets`. The datasets are loaded in entirety into the RAM and therefore is advised to be used only with sufficient amount of RAM. Support for lazy parallel loading in a `Channel` will be given soon. 

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

Added support for `RL Unplugged` atari datasets. The datasets that are stored in the form of `.tfrecord` are fetched into julia. Lazy loading with multi threading is implemented.

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
-[Features for Offline Reinforcement Learning Pipeline #359](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/discussions/359)
- [Fix record_type issue #24](https://github.com/JuliaReinforcementLearning/TFRecord.jl/pull/24)

## Implementation Details and Challenges Faced

The challenge that was faced during the first week was to chart out a direction for RLDatasets.jl. So, I had to research implementations of the pipeline in [d3rlpy](https://github.com/takuseno/d3rlpy), [TF.data.Dataset](https://www.tensorflow.org/datasets) etc. Then I narrowed down some of the inspiring ideas in the [discussion](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/discussions/359).

Later I made the [implementation](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/384) as a wrapper around d4rl python library which was discarded as it did not align with the purpose of the library. 

It was finalized that the package was going to be a lightweight implementation without much dependencies. So, we decided to use [DataDeps.jl](https://github.com/oxinabox/DataDeps.jl) for registering, tracking and locating datasets without any hassle. The thing that I learnt here was how to make a package, manage its dependencies and choose which package would be the right fit for the job. I also had to learn about Iterator interfaces in julia to convert the type that is output by the `dataset` function into an iterator. `d4rl-pybullet` was also implemented in a similar fashion.

Implementation of `Google Research Atari DQN Replay Datasets` was harder because it was quite a large dataset and even one shard didn't exactly fit into memory. One of the major things that I had to figure out was how the data was stored and how to retrieve it. Initially I planned to use `GZip.jl` to unpack the gzip files and use `NPZ.jl` to read the files. And NPZ wasn't able to read from `GZipStream` by itself so I had to adapt the functions in `NPZ` to read the stream. Later we decided to use `CodecZlib` to get a decompressed buffer channel output which was natively supported by `NPZ`. We also had to test it internally and skip the CI test because CI wouldn't be able to handle the dataset.

For supporting RL Unplugged dataset I had to learn about `.tfrecord` files, Protocol Buffers, Channels, buffered Channels and using multi threading in a lot of occasions that took a lot of time to learn, The final implementation was however based on already existing work in `TFRecord.jl`. 

Some of the more interesting pieces of code used in RL Unplugged dataset.

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
Multi threaded iteration over a Channel to `put!` into another Channel. While the implementation inside `TFRecord.read` is multi threaded in itself. Took quite a while for me to understand these nuances.

```julia
res = Channel{RLTransition}(n_preallocations; taskref=taskref, spawn=true) do ch
    Threads.@threads for i in 1:batch_size
        put!(ch, deepcopy(batch(buffer_template, popfirst!(transitions), i)))
    end
end
```
Multi threaded batching.

## Implications

Equipping RL.jl with RLDatasets.jl is a key step in making the package more industry relevant because different Offline algorithms can be compared with respect to a variety of standard offline dataset benchmarks.

## Future Plan

Enabling more feature that were mentioned in [Features for Offline Reinforcement Learning Pipeline #359](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/discussions/359) would be the next obvious step after an OPE method (like FQE) has been explored. Dataset generation, storage and policy parameter storage would be great for the package.