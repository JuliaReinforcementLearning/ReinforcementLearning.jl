# How to implement a new algorithm

All algorithms in ReinforcementLearning.jl are based on a common `run` function defined in [run.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/main/src/ReinforcementLearningCore/src/core/run.jl) that will be dispatched based on the type of its arguments. As you can see, the run function first performs a check and then calls a "private" `_run(policy::AbstractPolicy, env::AbstractEnv, stop_condition, hook::AbstractHook)`, this is the main function we are interested in. It consists of an outer and an inner loop that will repeateadly call `optimise!(policy, stage, env)`. 

Let's look at it closer in this simplified version (hooks are discussed [here](./How_to_use_hooks.md)):

```julia
function _run(policy::AbstractPolicy,
        env::AbstractEnv,
        stop_condition::AbstractStopCondition,
        hook::AbstractHook,
        reset_condition::AbstractResetCondition)

    push!(policy, PreExperimentStage(), env)
    is_stop = false
    while !is_stop
        reset!(env)
        push!(policy, PreEpisodeStage(), env)
        optimise!(policy, PreEpisodeStage())

        while !reset_condition(policy, env) # one episode
            push!(policy, PreActStage(), env)
            optimise!(policy, PreActStage())

            action = RLBase.plan!(policy, env)
            act!(env, action)

            push!(policy, PostActStage(), env)
            optimise!(policy, PostActStage())

            if check_stop(stop_condition, policy, env)
                is_stop = true
                push!(policy, PreActStage(), env)
                optimise!(policy, PreActStage())
                RLBase.plan!(policy, env)  # let the policy see the last observation
                break
            end
        end # end of an episode

        push!(policy, PostEpisodeStage(), env)  # let the policy see the last observation
        optimise!(policy, PostEpisodeStage())

    end
    push!(policy, PostExperimentStage(), env)
end

```

Implementing a new algorithm mainly consists of creating your own `AbstractPolicy` (or `AbstractLearner`, see [this section](#using-resources-from-rlcore)) subtype, its action sampling method (by overloading `Base.push!(policy::YourPolicyType, env)`) and implementing its behavior at each stage. However, ReinforcemementLearning.jl provides plenty of pre-implemented utilities that you should use to 1) have less code to write 2) lower the chances of bugs and 3) make your code more understandable and maintainable (if you intend to contribute your algorithm). 

## Using Agents
The recommended way is to use the policy wrapper `Agent`. An agent is itself an `AbstractPolicy` that wraps a policy and a trajectory (also called Experience Replay Buffer in RL literature). Agent comes with default implementations of `push!(agent, stage, env)` that will probably fit what you need at most stages so that you don't have to write them again. Looking at the [source code](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/main/src/ReinforcementLearningCore/src/policies/agent.jl/), we can see that the default Agent calls are  

```julia
function RLBase.plan!(agent::Agent{P,T,C}, env::AbstractEnv) where {P,T,C}
    action = RLBase.plan!(agent.policy, env)
    push!(agent.trajectory, agent.cache, action)
    action
end

function Base.push!(agent::Agent, ::PreActStage, env::AbstractEnv)
    push!(agent, state(env))
end

function Base.push!(agent::Agent{P,T,C}, ::PostActStage, env::E) where {P,T,C,E<:AbstractEnv}
    push!(agent.cache, reward(env), is_terminated(env))
end
```

 The first function, `RLBase.plan!(agent::Agent, env::AbstractEnv)`, is called at the `action = RLBase.plan!(policy, env)` line. It gets an action from the policy (since you implemented the `RLBase.plan!(your_new_policy, env)` function), then it pushes its `cache` and the action to the trajectory of the agent. Finally, it empties the cache and returns the action (which is immediately applied to env after). At the `PreActStage()` and the `PostActStage()`, the agent simply records the current state of the environment, the returned reward and the terminal state signal to its cache (to be pushed to the trajectory by the first function). Notice that the `PreActStage` push and plan are called at the end of the episode loop to store the very last state and actions in the trajectory. 

If you need a different behavior at some stages, then you can overload the `Base.push!(Agent{<:YourPolicyType}, [stage,] env)` or `Base.push!(Agent{<:Any, <: YourTrajectoryType}, [stage,] env)`, or `Base.plan!`, depending on whether you have a custom policy or just a custom trajectory. For example, many algorithms (such as PPO) need to store an additional trace of the logpdf of the sampled actions and thus overload the function at the `PreActStage()`.

## Updating the policy

Finally, you need to implement the learning function by implementing `RLBase.optimise!(::YourPolicyType, ::Stage, ::Trajectory)`. By default this does nothing at all stages. Overload it on the stage where you wish to optimise (most often, at `PreActStage()`, `PostActStage()` or `PostEpisodeStage()`). This function should loop the trajectory to sample batches. Inside the loop, put whatever is required. For example:

```julia
function RLBase.optimise!(p::YourPolicyType, ::PostEpisodeStage, traj::Trajectory)
    for batch in traj
        optimise!(p, batch)
    end
end

```
where `optimise!(p, batch)` is a function that will typically compute the gradient and update a neural network, or update a tabular policy. What is inside the loop is free to be whatever you need but it's a good idea to implement a `optimise!(p::YourPolicyType, batch::NamedTuple)` function for clarity instead of coding everything in the loop. This is further discussed in the next section on `Trajectory`s. An example of where this could be different is when you want to update priorities, see [the PER learner](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/main/src/ReinforcementLearningZoo/src/algorithms/dqns/prioritized_dqn.jl) for an example.

## ReinforcementLearningTrajectories

Trajectories are handled in a stand-alone package called [ReinforcementLearningTrajectories](https://github.com/JuliaReinforcementLearning/ReinforcementLearningTrajectories.jl). However, it is core to the implementation of your algorithm as it controls many aspects of it, such as the batch size, the sampling frequency, or the replay buffer length.
A `Trajectory` is composed of three elements: a `container`, a `controller`, and a `sampler`. 

### Container

The container is typically an `AbstractTraces`, an object that store a set of `Trace` in a structured manner. You can either define your own (and contribute it to the package if it is likely to be usable for other algorithms), or use a predefined one if it exists. 

The most common `AbstractTraces` object is the `CircularArraySARTSTraces`, this is a container of a fixed length that stores the following traces: `:state` (S), `:action` (A), `:reward` (R), `:terminal` (T), which together are aliased to `SART = (:state, :action, :reward, :terminal)`. Let us see how it is constructed in this simplified version as an example of how to build a custom trace. 

```julia
function (capacity, state_size, state_eltype, action_size, action_eltype, reward_eltype)
    MultiplexTraces{SS}(CircularArrayBuffer{state_eltype}(state_size..., capacity + 1)) +
    MultiplexTraces{AA′}(CircularArrayBuffer{action_eltype}(action_size..., capacity + 1)) +
    Traces(
        reward=CircularArrayBuffer{reward_eltype}(1, capacity),
        terminal=CircularArrayBuffer{Bool}(1, capacity),
    )
end
```
We can see it is composed (with the `+` operator) of two `MultiplexTraces` and a `Traces`. 
- `MultiplexTraces` is a special Trace that stores two names in one container. In this case, the two names of the first one are `SS′ = (:state, :next_state)`. When sampled for the `:next_state` at index `i`, it will return the state stored at `i+1`. This way, states and next states are managed together seamlessly (notice however that these must have +1 in their capacity). 
- `Traces` is for simpler traces, simply define a name (reward and terminal here) for each and assign them to a container.

The containers used here are `CircularArrayBuffers`. These are preallocated arrays that, once full, will overwrite the oldest element in storage, as if it was circular. It takes as arguments the size of each of its dimensions, where the last one is the capacity of the buffer. For example, if a state is a 256 x 256 image, `state_size` would be a tuple `(256,256)`. For vector states use `(256,)` and for scalars `1` or `()`. 

### Controller

ReinforcementLearningTrajectories' design aims to eventually support distributed experience collection, hence the somewhat involved design of trajectories and the presence of a controller. The controller is an object that will decide when the trajectory is ready to be sampled. Let us see with an example of the only controller so far: `InsertSampleRatioController(ratio, threshold)`. Despite its name, it is quite simple: this controller records the number of insertions (`ins`) in the trajectory and the number of batches sampled (`sam`); if `sam/ins > ratio` then the controller will stop the batch sample loop. For example, a ratio of 1/1000 means that one batch will be sampled every 1000 insertions in the trajectory. `threshold` is simply a minimum number of insertions required before the the controller starts sampling.

### Sampler

The sampler is the object that will fetch data in your trajectory to create the `batch` in the optimise for loop. The simplest one is the `BatchSampler{names}(batchsize, rng)`.`batchsize` is the number of elements to sample and `rng` is an optional argument that you may set to a custom rng for reproducibility. `names` is the set of traces the sampler must query. For example a `BatchSampler{(:state, :action, :next_state)}(32)` will sample a named tuple `(state = [32 states], action=[32 actions], next_state=[32 states that are one-off with respect that in state])`.

## Using resources from RLCore

RL algorithms typically only differ partially  but broadly use the same mechanisms. The subpackage RLCore contains some utilities that you can reuse to implement your algorithm.

### QBasedPolicy

`QBasedPolicy` is a policy that wraps a Q-Value _learner_ (tabular or approximated) and an _explorer_. Use this wrapper to implement a policy that directly uses a Q-value function to 
decide its next action. In that case, instead of creating an `AbstractPolicy` subtype for your algorithm, define an `AbstractLearner` subtype and specialize `RLBase.optimise!(::YourLearnerType, ::Stage, ::Trajectory)`. This way you will not have to code the interaction between your policy and the explorer yourself. 
RLCore provides the most common explorers (such as epsilon-greedy, UCB, etc.). 

### Neural and linear approximators

If your algorithm uses a neural network or a linear approximator to approximate a function trained with `Flux.jl`, use the `Approximator`. Approximator 
wraps a `Flux` model and an `Optimiser` (such as Adam or SGD). Your `optimise!(::PolicyOrLearner, batch)` function will probably consist in computing a gradient 
and call the `RLCore.optimise!(app::Approximator, gradient::Flux.Grads)` after that. 

Common model architectures are also provided such as the `GaussianNetwork` for continuous policies with diagonal multivariate policies; and `CovGaussianNetwork` for full covariance (very slow on GPUs at the moment).

### Utils
In utils/distributions.jl you will find implementations of gaussian log probabilities functions that are both GPU compatible and differentiable and that do not require the overhead of using Distributions.jl structs.

## Conventions
Finally, there are a few "conventions" and good practices that you should follow, especially if you intend to contribute to this package (don't worry we'll be happy to help if needed).
 
### Random Numbers
ReinforcementLearning.jl aims to provide a framework for reproducible experiments. To do so, make sure that your policy type has a `rng` field and that all random operations (e.g. action sampling) use `rand(your_policy.rng, args...)`. For trajectory sampling, you can set the sampler's rng to that of the policy when creating and agent or simply instantiate its own rng.

### GPU compatibility
Deep RL algorithms are often much faster when the neural nets are updated on a GPU. For now, we only support CUDA.jl as a backend. This means that you will have to think about the transfer of data between the CPU (where the trajectory is) and the GPU memory (where the neural nets are). To do so you will find in utils/device.jl some functions that do most of the work for you. The ones that you need to know are `send_to_device(device, data)` that sends data to the specified device, `send_to_host(data)` which sends data to the CPU memory (it fallbacks to `send_to_device(Val{:cpu}, data)`) and `KernelAbstractions.get_backend(x)` that returns the device on which `x` is. 
Normally, you should be able to write a single implementation of your algorithm that works on CPU and GPUs thanks to the multiple dispatch offered by Julia.

GPU friendlyness will also require that your code does not use _scalar indexing_ (see the CUDA.jl documentation for more information), make sure to test your algorithm on the GPU after disallowing scalar indexing by using `CUDA.allowscalar(false)`.

Finally, it is a good idea to implement the `Flux.gpu(yourpolicy)` and `cpu(yourpolicy)` functions, for user convenience. Be careful that sampling on the GPU requires a specific type of rng, you can generate one with `CUDA.default_rng()`
