# How to implement a new algorithm

All algorithms in ReinforcementLearning.jl are based on a common `run` function defined in [run.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/main/src/ReinforcementLearningCore/src/core/run.jl) that will be dispatched based on the type of its arguments. As you can see, the run function first performs a check and then calls a "private" `_run(policy::AbstractPolicy, env::AbstractEnv, stop_condition, hook::AbstractHook)`, this is the main function we are interested in. It consists of an outer and an inner loop that will repeateadly call `policy(stage, env)`. 

Let's look at it closer in this simplified version (hooks are discussed [here](./How_to_use_hooks.md)):

```julia
function _run(policy::AbstractPolicy, env::AbstractEnv, stop_condition, hook::AbstractHook)

    policy(PreExperimentStage(), env)
    is_stop = false
    while !is_stop
        reset!(env)
        policy(PreEpisodeStage(), env)

        while !is_terminated(env) # one episode
            policy(PreActStage(), env)
            env |> policy |> env
            optimise!(policy)
            policy(PostActStage(), env)
            if stop_condition(policy, env)
                policy(PreActStage(), env)
                policy(env)
                is_stop = true
                break
            end
        end # end of an episode

        if is_terminated(env)
            policy(PostEpisodeStage(), env)
        end
    end
end
```

Implementing a new algorithm mainly consists of creating your own `AbstractPolicy` subtype, its action sampling function `(policy)(env)` and implementing its behavior at each stage. However, ReinforcemementLearning.jl provides plenty of pre-implemented utilities that you should use to 1) have less code to write 2) lower the chances of bugs and 3) make your code more understandable and maintainable (if you intend to contribute your algorithm). 

## Using Agents
A better way is to use the policy wrapper `Agent`. An agent is an AbstractPolicy that wraps a policy and a trajectory (also called Experience Replay Buffer in RL literature). Agent comes with default implementations of `agent(stage, env)` that will probably fit what you need at most stages so that you don't have to write them again. Looking at the [source code](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/main/src/ReinforcementLearningCore/src/policies/agent.jl/), we can see that the default Agent calls are  

```julia
function (agent::Agent)(env::AbstractEnv)
    action = agent.policy(env)
    push!(agent.trajectory, (agent.cache..., action = action))
    agent.cache = (;)
    action
end

(agent::Agent)(::PreActStage, env::AbstractEnv) =
    agent.cache = (agent.cache..., state = state(env))

(agent::Agent)(::PostActStage, env::AbstractEnv) =
    agent.cache = (agent.cache..., reward = reward(env), terminal = is_terminated(env))
```

The default behavior at other stages is a no-op. The first function, `(agent::Agent)(env::AbstractEnv)`, is called at the `env |> policy |> env` line. It gets an action from the policy (since you implemented the `your_new_policy(env)` function), then it pushes its `cache` and the action to the trajectory of the agent. Finally, it empties the cache and returns the action (which is immediately applied to env after). At the `PreActStage()` and the `PostActStage`, the agent simply records the current state of the environment, the returned reward and the terminal state signal to its cache (to be pushed to the trajectory by the first function).

If you need a different behavior at some stages, then you can overload the `(Agent{<:YourPolicyType})([stage,] env)` or (Agent{<:Any, <: YourTrajectoryType})([stage,] env), depending on whether you have a custom policy or just a custom trajectory. For example, many algorithms (such as PPO) need to store an additional trace of the logpdf of the sampled actions and thus overload the function at the `PreActStage()`.

## Updating the policy

Finally, you need to implement the learning function by implementing `RLBase.optimise!(p::YourPolicyType, batch::NamedTuple)` (see that it is called by `optimise!(agent)` then `RLBase.optimise!(p::YourPolicyType, b::Trajectory)`). 
In principle you can do the update at other stages by overload the `(agent::Agent)` but this is not recommended because the trajectory may not be consistent and samples could be incorrect. If you choose to do it, make sure to know what you are doing. 

## ReinforcementLearningTrajectories

Trajectories are handled in a stand-alone package called [ReinforcementLearningTrajectories](https://github.com/JuliaReinforcementLearning/ReinforcementLearningTrajectories.jl). Refer to its documentation (in progress) to learn how to use it.

## Using resources from RLCore

RL algorithms typically only differ partially  but broadly use the same mechanisms. The subpackage RLCore contains a lot of utilities that you can reuse to implement your algorithm.

The utils folder contains utilities and extensions to external packages to fit needs that are specific to RL.jl. We will not list them all here, but it is a good idea to skim over the files to see what they contain. The policies folder notably contains several explorer implementations. Here are a few interesting examples:

- `QBasedPolicy` wraps a policy that relies on a Q-Value _learner_ (tabular or approximated) and an _explorer_ . 
RLCore provides several pre-implemented learners and the most common explorers (such as epsilon-greedy, UCB, etc.). 

- If your algorithm use tabular learners, check out the tabular_learner.jl and the tabular_approximator source files. If your algorithms uses deep neural nets then use the `NeuralNetworkApproximator` to wrap an Neural Network and an optimizer. Common policy architectures are also provided such as the `GaussianNetwork`.

- Equivalently, the `VBasedPolicy` learner is provided for algorithms that use a state-value function. Though they are not bundled in the same folder, most approximators can be used with a VBasedPolicy too.

<!--- ### Batch samplers
 Since this is going to be outdated soon, I'll write this part later on when Trajectories.jl will be done -->

- In utils/distributions.jl you will find implementations of gaussian log probabilities functions that are both GPU compatible and differentiable and that do not require the overhead of using Distributions.jl structs.

## Conventions
Finally, there are a few "conventions" and good practices that you should follow, especially if you intend to contribute to this package (don't worry we'll be happy to help if needed).
 
### Random Numbers
ReinforcementLearning.jl aims to provide a framework for reproducible experiments. To do so, make sure that your policy type has a `rng` field and that all random operations (e.g. action sampling or trajectory sampling) use `rand(your_policy.rng, args...)`.

### GPU friendlyness
Deep RL algorithms are often much faster when the neural nets are updated on a GPU. For now, we only support CUDA.jl as a backend. This means that you will have to think about the transfer of data between the CPU (where the trajectory is) and the GPU memory (where the neural nets are). To do so you will find in utils/device.jl some functions that do most of the work for you. The ones that you need to know are `send_to_device(device, data)` that sends data to the specified device, `send_to_host(data)` which sends data to the CPU memory (it fallbacks to `send_to_device(Val{:cpu}, data)`) and `device(x)` that returns the device on which `x` is. 
Normally, you should be able to write a single implementation of your algorithm that works on CPU and GPUs thanks to the multiple dispatch offered by Julia.

GPU friendlyness will also require that your code does not use _scalar indexing_ (see the CUDA.jl documentation for more information), make sure to test your algorithm on the GPU after disallowing scalar indexing by using `CUDA.allowscalar(false)`.

Finally, it is a good idea to implement the `Flux.gpu(yourpolicy)` and `cpu(yourpolicy)` functions, for user convenience. Be careful that sampling on the GPU requires a specific type of rng, you can generate one with `CUDA.default_rng()`
