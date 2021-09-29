@def title = "General Pipeline for Offline Reinforcement Learning Evaluation Report"
@def description = """
    This is a technical report of the Summer OSPP project [Establish a General Pipeline for Offline Reinforcement Learning Evaluation](https://summer.iscas.ac.cn/#/org/prodetail/210370741?lang=en) used for final-term evaluation.
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
        "publishedDate":"2021-09-29",
        "citationText":"Prasidh Srikumar, 2021"
    }"""

@def appendix = """
    ### Corrections
    If you see mistakes or want to suggest changes, please [create an issue](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues) in the source repository.
    """

@def bibliography = "bibliography.bib"

# Technical Report
The following is the final term evaluation report of "General Pipeline for Offline Reinforcement Learning Evaluation Report" in OSPP. Details of all the work that has been done after the mid-term evaluation and some explanation on the current status of the Package are given. Some exciting work that is possible based on this project is also given.

## Project Information

#### Project Name
General Pipeline for Offline Reinforcement Learning Evaluation

#### Background
With more advancement happening in Offline Reinforcement Learning in recent years and more availability of data, it is vital to implement Offline RL algorithms in RL.jl. Besides, It is also important for Offline Reinforcement Learning algrithms to have Datasets that helps in benchmarking algorithms and policies that are useful in benchmarking Off Policy Evaluation Methods. Off Policy Evaluation is important for reliable algorithms to be developed in a completely Offline setting. This project aims at providing reliable datasets and policies that are based on the latest research in the field of Offline Reinforcement Learning.

#### Objectives
- Create a package called **ReinforcementLearningDatasets.jl** that would aid in loading various standard datasets and policies that are available.
- Test out and implement an off policy evaluation algorithm like FQE on RLZoo to test on the available datasets and policies.

#### Time Planning

|     Date      |                                                                     Goals                                                                     |
| :-----------: | :-------------------------------------------------------------------------------------------------------------------------------------------: |
| 07/01 - 07/14 |               Brainstorm various ideas that are possible for the implementation of RLDatasets.jl and finalize the key features.               |
| 07/15 - 07/20 |                                     Made a basic julia wrapper for `d4rl` environments and add some tests                                     |
| 07/21 - 07/30 |                                                Implemented `d4rl` and `d4rl-pybullet` datasets                                                |
| 07/31 - 08/06 |                                              Implemented  `Google Research DQN Replay Datasets`                                               |
| 08/07 - 08/14 | Implemented `RL Unplugged atari datasets`, setup the docs, added README.md. Made the package more user friendly. Make the **mid-term report** |
| 08/15 - 08/30 |                  Add bsuite datasets, polish the interface, finalize the structure of the codebase. Fix problem with windows                  |
| 09/01 - 09/15 |         Add support for policy loading from [Benchmarks for Deep Off-Policy Evaluation](https://github.com/google-research/deep_ope)          |
| 09/16 - 09/30 |                   Research about OPE methods, implement FQE and test basic performance. Complete the **final-term report**                    |

There are some changes to the original timeline planned but the basic objectives of the project are accomplished.

## Completed Work
The following work has been done post mid-term evaluation.

### Summary
The following is the summary of the project work.

- Polished and finalized the structure of the package. Improved usability by updating the [docs](https://juliareinforcementlearning.org/docs/rldatasets/) accordingly.
- Fixed the `run` error that was shown in windows.
- Added `Bsuite` and all `DM` environements including [`DeepMind Control Suite Dataset`](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged#deepmind-control-suite-dataset), [`DeepMind Lab Dataset`](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged#deepmind-lab-dataset) and [`DeepMind Locomotion Dataset`](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged#deepmind-locomotion-dataset_) in RL Unplugged Datasets\dcite{DBLP:journals/corr/abs-2006-13888}.
- Added [Deep OPE](https://github.com/google-research/deep_ope)\dcite{DBLP:journals/corr/abs-2103-16596} models for D4RL datasets.
- Researched and implemented FQE\dcite{DBLP:journals/corr/abs-2007-09055} for which the basic implementation works. There are some flaws that need to be fixed.


### Bsuite Datasets
It involved work similar to RL Unplugged Atari Datasets which involves multi threaded dataloading. It is implemented using a [`Ring Buffer`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/f1837a93c4c061925d92167c3480a423007dae5c/src/ReinforcementLearningDatasets/src/rl_unplugged/util.jl#L89) for storing and loading batches of data.

#### Ring Buffer

Huge thanks to [Jun Tian](https://github.com/findmyway) for the implementation.

The RingBuffer is based on having two `Channel`s, one for holding the buffer that contains empty batches (`buffers`) that can be used later for making batches with data. The `results` `Channel` is used for holding batches with data. The `current` holds the latest result.

```julia
mutable struct RingBuffer{T} <: AbstractChannel{T}
    buffers::Channel{T}
    current::T
    results::Channel{T}
end
```
The `RingBuffer` is created using the following code. It creates an empty `buffers` `Channel` that is then used to fill up `results` by performing inplace operations for making batches with data.

```julia
function RingBuffer(f!, buffer::T;sz=Threads.nthreads(), taskref=nothing) where T
    buffers = Channel{T}(sz)
    for _ in 1:sz
        put!(buffers, deepcopy(buffer))
    end
    results = Channel{T}(sz, spawn=true, taskref=taskref) do ch
        Threads.foreach(buffers;schedule=Threads.StaticSchedule()) do x
        # for x in buffers
            f!(x)  # in-place operation
            put!(ch, x)
        end
    end
    RingBuffer(buffers, buffer, results)
end
```

Whenever a batch is taken from the buffer, the following code gets called.

```julia
function Base.take!(b::RingBuffer)
    put!(b.buffers, b.current)
    b.current = take!(b.results)
    b.current
end
```

I would have implemented a simpler one channel buffer, but `RingBuffer` proved to be more effective.

#### Implementation
The files are read and the datapoints are put in a `Channel`.
```julia
ch_src = Channel{BSuiteRLTransition}(n * tf_reader_sz) do ch
    for fs in partition(files, n)
        Threads.foreach(
            TFRecord.read(
                fs;
                compression=:gzip,
                bufsize=tf_reader_bufsize,
                channel_size=tf_reader_sz,
            );
            schedule=Threads.StaticSchedule()
        ) do x
            put!(ch, BSuiteRLTransition(x, game))
        end
    end
end
```
The datapoints are then put in a `RingBuffer` which is returned.
```julia
res = RingBuffer(buffer;taskref=taskref, sz=n_preallocations) do buff
    Threads.@threads for i in 1:batch_size
        batch!(buff, take!(transitions), i)
    end
end
```
#### Working
The `bsuite_params` function can be used to get the possible arguments can be passed into the function.
```julia
julia> bsuite_params()
┌ Info: ["cartpole", "catch", "mountain_car"]
│   shards = 0:4
└   type = 3-element Vector{String}: …
```

To get the dataset `rl_unplugged_bsuite_dataset` function can be called.
```julia
julia> rl_unplugged_bsuite_dataset("cartpole", [1], "full")
Progress: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:06
RingBuffer{ReinforcementLearningDatasets.BSuiteRLTransition}(Channel{ReinforcementLearningDatasets.BSuiteRLTransition}(12), ReinforcementLearningDatasets.BSuiteRLTransition(Float32[1.0f-45 1.0f-45 … NaN 0.0; 0.0 0.0 … NaN 0.0; … ; 3.0f-45 4.0f-45 … NaN 9.0f-44; 0.0 0.0 … NaN 0.0], [140269590665616, 140269590665616, 140269590665616, 
...
julia> take!(ds)
ReinforcementLearningDatasets.BSuiteRLTransition(Float32[-0.3289344 0.26131696 … -0.015311318 0.49089232; 0.31783995 -1.2033445 … 0.04303875 -0.24614102; … ; -0.27250051 1.0421202 … -0.17690773 0.2694671; 0.697 0.805 … 0.009 0.955], [0, 0, 0, 0, 0, 1, 1, 0, 0, 1  …  2, 0, 1, 0, 2, 0,
...
```
### DM Datasets

The DM datasets load and work similarly to Bsuite datasets. Since, I made one file to manage `DM Control`, `DM Lab` and `DM Locomotion`, there had to be a lot of post processing work to handle all the edge cases presented by each of the dataset.

The types also had to be created based on the individual datasets so that the code is good at loading efficiently.

#### Example of type handling used.
```julia
function make_transition(example::TFRecord.Example, feature_size::Dict{String, Tuple})
    f = example.features.feature
    
    observation_dict = Dict{Symbol, AbstractArray}()
    next_observation_dict = Dict{Symbol, AbstractArray}()
    transition_dict = Dict{Symbol, Any}()

    for feature in keys(feature_size)
        if split(feature, "/")[1] == "observation"
            ob_key = Symbol(chop(feature, head = length("observation")+1, tail=0))
            if split(feature, "/")[end] == "egocentric_camera"
                cam_feature_size = feature_size[feature]
                ob_size = prod(cam_feature_size)
                observation_dict[ob_key] = reshape(f[feature].bytes_list.value[1][1:ob_size], cam_feature_size...)
                next_observation_dict[ob_key] = reshape(f[feature].bytes_list.value[1][ob_size+1:end], cam_feature_size...)
            else
                if feature_size[feature] == ()
                    observation_dict[ob_key] = f[feature].float_list.value
                else
                    ob_size = feature_size[feature][1]
                    observation_dict[ob_key] = f[feature].float_list.value[1:ob_size]
                    next_observation_dict[ob_key] = f[feature].float_list.value[ob_size+1:end]
                end
            end
        elseif feature == "action"
            ob_size = feature_size[feature][1]
            action = f[feature].float_list.value
            transition_dict[:action] = action[1:ob_size]
            transition_dict[:next_action] = action[ob_size+1:end]
        elseif feature == "step_type"
            transition_dict[:terminal] = f[feature].float_list.value[1] == 2
        else
            ob_key = Symbol(feature)
            transition_dict[ob_key] = f[feature].float_list.value[1]
        end
    end
    state_nt = (state = NamedTuple(observation_dict),)
    next_state_nt = (next_state = NamedTuple(next_observation_dict),)
    transition = NamedTuple(transition_dict)

    merge(transition, state_nt, next_state_nt)
end
```
The batch is made based on the internal types that are available based on the specific dataset.

#### Usage
```julia
julia> rl_unplugged_dm_dataset("fish_swim", [1]; type="dm_control_suite")
Progress: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:02
RingBuffer{NamedTuple{(:reward, :episodic_reward, :discount, :state, :next_state, :action, :next_action, :terminal), Tuple{Vector{Float32}, Vector{Float32}, Vector{Float32}, NamedTuple{(:joint_angles, :upright, :target, :velocity), NTuple{4, Matrix{Float32}}}, NamedTuple{(:joint_angles, :upright, :target, :velocity), NTuple{4, Matrix{Float32}}}, Matrix{Float32}, Matrix{Float32}, Vector{Bool}}}}(Channel{NamedTuple{(:reward, :episodic_reward, :discount, :state, :next_state, :action, :next_action, :terminal), Tuple{Vector{Float32}, Vector{Float32}, Vector{Float32}, NamedTuple{(:joint_angles, :upright, :target, :velocity), NTuple{4, Matrix{Float32}}}, NamedTuple{(:joint_angles, :upright, :target, :velocity), NTuple{4, Matrix{Float32}}}, Matrix{Float32}, Matrix{Float32}, Vector{Bool}}}}(12), (reward = Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.4009038f-36, 4.5764f-41  …  0.0, 0.0, 2.3634088f-28, 4.5765f-41, 1.1692183f-34, 4.5764f-41, 0.0, 0.0,
...
```
### Deep OPE

Support is given for D4RL policies provided in [Deep OPE](https://github.com/google-research/deep_ope).

#### Implementation
The policies that are given [here](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningDatasets/src/deep_ope/d4rl/d4rl_policies.jl) are loaded using `d4rl_policy` function.

The policies are loaded into a `D4RLGaussianNetwork` which will be integrated into `GaussianNetwork` in RLCore soon.
```julia
Base.@kwdef struct D4RLGaussianNetwork{P,U,S}
    pre::P = identity
    μ::U
    logσ::S
end
```

The network returns the `a`, `μ` based on the parameters that are passed into it.

```julia
function (model::D4RLGaussianNetwork)(
    state::AbstractArray;
    rng::AbstractRNG=MersenneTwister(123), 
    noisy::Bool=true
)
    x = model.pre(state)
    μ, logσ = model.μ(x), model.logσ(x)
    if noisy
        a = μ + exp.(logσ) .* Float32.(randn(rng, size(μ)))
    else
        a = μ + exp.(logσ)
    end
    a, μ
end 
```
The weights are loaded using the following [code](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningDatasets/src/deep_ope/d4rl/d4rl_policy.jl).

To know the real life performance of the networks an auxillary function `deep_ope_d4rl_evaluate` is also given which gives the unicode plot showing the performance of the policy. The code is given [here](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningDatasets/src/deep_ope/d4rl/evaluate.jl).

#### Working
The params needed for loading the policies can be obtained using `d4rl_policy_params`

```julia
julia> d4rl_policy_params()
┌ Info: Set(["relocate", "maze2d_large", "antmaze_umaze", "hopper", "pen", "antmaze_medium", "walker", "hammer", "antmaze_large", "maze2d_umaze", "maze2d_medium", "ant", "door", "halfcheetah"])
│   agent = 2-element Vector{String}: …
└   epoch = 0:10
```

Sometimes the expected policy may not be available. So, it is always better to check which ones are available using `ReinforcementLearningDatasets.D4RL_POLICIES`.

```julia
julia> policy = d4rl_policy("hopper", "online", 3)
D4RLGaussianNetwork{Flux.Chain{Tuple{Flux.Dense{typeof(NNlib.relu), Matrix{Float32}, Vector{Float32}}, Flux.Dense{typeof(NNlib.relu), Matrix{Float32}, Vector{Float32}}}}, Flux.Chain{Tuple{Flux.Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}}}, Flux.Chain{Tuple{Flux.Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}}}}(Chain(Dense(11, 256, relu), Dense(256, 256, relu)), Chain(Dense(256, 3)), Chain(Dense(256, 3)))
```

The policy will return the `a` and `μ` for a state that is given.

`deep_ope_d4rl_evaluate` is a helper function that helps visualise the performance of the agent. The more the epoch number, the better the performance.
```julia
julia> deep_ope_d4rl_evaluate("halfcheetah", "online", 3)
Progress: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:01
                    halfcheetah-medium-v0 scores
              +----------------------------------------+       
         8100 |                                       .| scores
              |                                      .'|       
              |        .                           .'  |       
              |      .' :                         .'   |       
              |    .'   :       :.               :     |       
              |  .'      :     .' :            .'      |       
              |.'        '.    :   :          .'       |       
   score      |           :   :     '.       .'        |       
              |           '. .'      '.      :         |       
              |            : :        :     :          |       
              |            ':          :   :           |       
              |                         : .'           |       
              |                         '.:            |       
              |                          '             |       
         7400 |                                        |       
              +----------------------------------------+       
              1                                       10
                               episode
julia> deep_ope_d4rl_evaluate("halfcheetah", "online", 10)
Progress: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:01
                     halfcheetah-medium-v0 scores
               +----------------------------------------+       
         12000 |                 .                      | scores
               |''''..........''''.       :.       :''''|       
               |                  :       :'.     :     |       
               |                  :      .' '.   :      |       
               |                  '.     :   :  :       |       
               |                   :     :    ::        |       
               |                   :    .'     '        |       
   score       |                   '.   :               |       
               |                    :   :               |       
               |                    :  .'               |       
               |                    '. :                |       
               |                     : :                |       
               |                     :.'                |       
               |                     ::                 |       
          2000 |                      :                 |       
               +----------------------------------------+       
               1                                       10
                                episode
```
### FQE
A major amount of time was spent on researching about OPE methods of which `FQE` was the most appropriate given that the use case is Deep Reinforcement Learning.

[Batch Policy Learning under Constraints](https://arxiv.org/pdf/1903.08738.pdf)\dcite{DBLP:journals/corr/abs-1903-08738} introduces the FQE and uses it for offline reinforcement learning under constraints and achieves remarkable results by calculating new constraint cost functions with the datasets. The algorithm that is implemented is similar to the one that is proposed here.

\dfig{body;FQE_Original.png}

The implementation in RLZoo is based on [Hyperparameter Selection for Offline Reinforcement Learning](https://arxiv.org/pdf/2007.09055.pdf)\dcite{DBLP:journals/corr/abs-2007-09055}. This is very similar to the algorithm that we discussed earlier. The paper uses OPE as a method for offline hyper paramater selection.

\dfig{body;OPE_and_Online_Hyperparameter_Selection.png}

The average of values chosen by the policies based on initial states can be taken as the reward that the policy would gain from the environment. So, the same can be used for online hyper parameter selection.

The pseudocode for the implementation and the objective function are as follows.

\dfig{body;FQE_Impl.png}

\dfig{body;FQE_Objective.png}

#### Implementation
Function parameters for the implemenation.

```julia
mutable struct FQE{
    P<:GaussianNetwork,
    C<:NeuralNetworkApproximator,
    C_T<:NeuralNetworkApproximator,
    R<:AbstractRNG,
 } <: AbstractLearner
    policy::P
    q_network::C
    target_q_network::C_T
    n_evals::Int
    γ::Float32
    batch_size::Int
    update_freq::Int
    update_step::Int
    tar_update_freq::Int
    rng::R
    #logging
    loss::Float32
end
```

The update function for the learner is a simple critic update based on the following procedure.

```julia
function RLBase.update!(l::FQE, batch::NamedTuple{SARTS})
    policy = l.policy
    Q, Qₜ = l.q_network, l.target_q_network

    D = device(Q)
    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    γ = l.γ
    batch_size = l.batch_size

    loss_func = Flux.Losses.mse

    q′ = Qₜ(vcat(s′, policy(s′)[1])) |> vec

    target = r .+ γ .* (1 .- t) .* q′

    gs = gradient(params(Q)) do
        q = Q(vcat(s, reshape(a, :, batch_size))) |> vec
        loss = loss_func(q, target)
        Zygote.ignore() do
            l.loss = loss
        end
        loss
    end
    Flux.Optimise.update!(Q.optimizer, params(Q), gs)

    if l.update_step % l.tar_update_freq == 0
        Qₜ = deepcopy(Q)
    end
end
```
#### Results
The [implementation](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/515) is still a work in progress because of some sampling error. But the algorithm that I implemented without RL.jl framework works as expected. 

##### Parameter Values:
-  Policy => CRR Policy
-  Env => PendulumEnv
-  q_networks => Two 64 neuron layers with `n_s+n_a` input neurons and `1` output neuron.
-  optimizer => ADAM(0.005)
-  loss => Flux.Losses.mse
-  γ => 0.99
-  batch_size = 256
-  update_freq, update_step = 1
-  tar_update_freq = 256

##### Evaluation Results

/dfig{body;FQE_Evaluation_Result.png}

mean=-243.0258f0

##### Actual Values

/dfig{body;Actual_Evaluation_Result.png}

mean = -265.7068139137983

### Relevant Commits and PRs

- [Fix RLDatasets.jl documentation (#467)](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commit/b29c9f01240d6aae9e6f7acc28a0a1e95cf29f76#diff-d7a7b3de8d5eedecb629c4d80b6b249d68d15d6f66a7ef768bf4eb937fd5a5d7)
- [Add bsuite datasets (#482)](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commit/4326df59296a6edc488b77f29c4968853280db85#diff-d7a7b3de8d5eedecb629c4d80b6b249d68d15d6f66a7ef768bf4eb937fd5a5d7)
- [Add dm datasets (#495)](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commit/9185c8548197dd4a6ef0cd7c84c3531c491e6447#diff-d7a7b3de8d5eedecb629c4d80b6b249d68d15d6f66a7ef768bf4eb937fd5a5d7)
- [Add support for deep ope in RLDatasets.jl (#500)](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commit/1a00766e9df3edc19cd7377a595b4563261a0356#diff-d7a7b3de8d5eedecb629c4d80b6b249d68d15d6f66a7ef768bf4eb937fd5a5d7)
- [WIP to implement FQE #515](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/515)

### Conclusion
The foundations of RLDatasets.jl package has been laid during the course of the project. The basic datasets except for Real World Datasets from RL Unplugged have been supported. Furthermore, D4RL policies have been successfully loaded and tested. The algorithm for FQE has been tried out with a minor implementation detail pending. 

With the completion of FQE the four requirements of OPE as laid out by [Deep OPE](https://github.com/google-research/deep_ope)\dcite{DBLP:journals/corr/abs-2103-16596} will be completed for D4RL.

\dfig{body;OPE_Requirements.png}

#### Future Scope
There are several exciting work that are possible from this point.

- Testing and improvement of already existing Offline Algorithms in RLZoo.jl.
- Integrating the existing RLDatasets.jl package to work well with RL.jl.
- Implementing more OPE algorithms proposed in [Empirical Study of Off-Policy Policy Evaluation for Reinforcement Learning paper](https://arxiv.org/pdf/1911.06854.pdf)\dcite{DBLP:journals/corr/abs-1911-06854} for use in Deep RL and Tabular RL.
- Implementation of other FQE methods like DiscreteFQE, [FQE-L2 (Statistical Bootstrapping for Uncertainty Estimation in Off-Policy Evaluation)](https://arxiv.org/pdf/2007.13609.pdf)\dcite{DBLP:journals/corr/abs-2007-13609}.
- Adding standard difficult benchmarks for existing Offline RL methods.
- Adding environments to work out of the box for evaluation OPE methods.
- Adding Scikit learn like [features](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/discussions/359) on top of RLDataset.jl. 