# How to implement a new algorithm

All algorithms in ReinforcementLearning.jl are based on a common `run` function defined in [run.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningCore/src/core/run.jl) that will be dispatched based on the type of its arguments. As you can see, the run function first performs a check and then calls a "private" `_run(policy::AbstractPolicy, env::AbstractEnv, stop_condition, hook::AbstractHook)`, this is the main function we are interested in. It consists of an outer and an inner loop that will repeateadly call `policy(stage, env [,action])`. 

Let's look at it closer in this simplified version (hooks are discussed [here](./How_to_use_hooks.md)):

```julia
function _run(policy::AbstractPolicy, env::AbstractEnv, stop_condition, hook::AbstractHook)

    policy(PRE_EXPERIMENT_STAGE, env)
    is_stop = false
    while !is_stop
        reset!(env)
        policy(PRE_EPISODE_STAGE, env)

        while !is_terminated(env) # one episode
            action = policy(env)

            policy(PRE_ACT_STAGE, env, action)

            env(action)

            policy(POST_ACT_STAGE, env)

            if stop_condition(policy, env)
                is_stop = true
                break
            end
        end # end of an episode

        if is_terminated(env)
            policy(POST_EPISODE_STAGE, env)
        end
    end
end
```

Implementing a new algorithm mainly consists of creating your own `AbstractPolicy` subtype, its action sampling function `(policy)(env)` and implementing its behavior at each stage. However, ReinforcemementLearning.jl provides plenty pre-implemented utilities that you should use to 1) have less code to write 2) lower the chances of bugs and 3) make your code more understandable and maintainable (if you intend to contribute your algorithm). 

## Using Agents
A better way is to use the policy wrapper `Agent`. An agent is an AbstractPolicy that wraps a policy and a trajectory (also called Experience Replay Buffer in RL literature). Agent comes with default implementations of `Agent(stage, agent, env)` that will probably fit what you need at most stages so that you don't have to write them again. Looking at the [source code](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningCore/src/policies/agents/agent.jl/), we can see that the default Agent calls are  

```julia
function (agent::Agent)(stage::AbstractStage, env::AbstractEnv [, action])
    update!(agent.trajectory, agent.policy, env, stage [,action])
    update!(agent.policy, agent.trajectory, env, stage)
end
```

Which consists of updating the trajectory then the policy. `update!(agent.policy, agent.trajectory, env, stage)` is a no-op by default at every stage but `update!(agent.trajectory, agent.policy, env, stage [,action])` comes with predefined updates that can be summarized as follows (keep in mind that trajectories are undergoing major changes and this will soon be outdated.). 

1. At the PRE\_ACT\_STAGE (after having sampled the action), add the current state and the sampled action to the trajectory. If your policy uses an action mask, it will also save it to a respective trace.
2. At the POST\_ACT\_STAGE (after having exerted the action to the environment), add the returned reward and save whether the new state is terminal.
3. At the POST\_EPISODE\_STAGE (before reseting the environment), save the last state to the trajectory. 
4. At the PRE\_EPISODE\_STAGE (after reseting the environment), remove the last state from the trajectory. 

If you need a different behavior for trajectories, then you may overload the `update!` function with your policy type or a custom trajectory type. For example, many algorithms (such as PPO) need to store an additional trace of the logpdf of the sampled actions and thus overload the function at the PRE\_ACT\_STAGE.

## Updating the policy

Finally, you need to implement the learning function by implementing `(your_policy)( env, stage)` or `update!(your_policy, trajectory, env, stage)`. This is usually done at the PRE\_ACT\_STAGE or the POST\_EPISODE\_STAGE, depending on the algorithm. It is not recommended to do it at other stages because the trajectory will not be consistent and samples from it will be be incorrect.  

## Using resources from RLCore

### Learners

RL algorithms typically differ partially but broadly use the same mechanisms. The subpackage RLCore contains a lot of utilities that you can reuse to implement your algorithm. These are implemented as types that you can impose on certain fields of your own policy type.

`QBasedPolicy` wraps a policy that relies on a Q-Value _learner_ (tabular or approximated) and an _explorer_ . 
RLCore provides several pre-implemented learners and the most common explorers (such as epsilon-greedy, UCB, etc.). 

If your algorithm use tabular learners, check out the tabular_learner.jl and the tabular_approximator source files. If your algorithms uses deep neural nets then use the `NeuralNetworkApproximator` to wrap an Neural Network and an optimizer. Common policy architectures are also provided such as the `GaussianNetwork`.

Equivalently, the `VBasedPolicy` learner is provided for algorithms that use a state-value function. Though they are not bundled in the same folder, most approximators can be used with a VBasedPolicy too.

<!--- ### Batch samplers
 Since this is going to be outdated soon, I'll write this part later on when Trajectories.jl will be done -->


### Extensions

The extensions folder contains extensions to external packages to fit needs that are specific to RL.jl. Notably, in the Distributions.jl you will find implementations of gaussian log probabilities functions that are both GPU compatible and differentiable and that do not require the overhead of using Distributions.jl structs.

## Conventions
Finally, there are a few "conventions" and good practices that you should follow, especially if you intend to contribute to this package (don't worry we'll be happy to help if needed).
 
### Random Numbers
ReinforcementLearning.jl aims to provide a framework for reproducible experiments. To do so, make sure that your policy type has a `rng` field and that all random operations (e.g. action sampling or trajectory sampling) use `rand(your_policy.rng, args...)`.

### GPU friendlyness
Deep RL algorithms are often much faster when the neural nets are updated on a GPU. For now, we only support CUDA.jl as a backend. This means that you will have to think about the transfer of data between the CPU (where the trajectory is) and the GPU memory (where the neural nets are). To do so you will find in extensions some functions that do most of the work for you. The ones that you need to know are `send_to_device(device, data)` that sends data to the specified device, `send_to_host(data)` which sends data to the CPU memory (it fallbacks to `send_to_device(Val{:cpu}, data)`) and `device(x)` that returns the device on which `x` is. 
Normally, you should be able to write a single implementation of your algorithm that works on CPU and GPUs thanks to the multiple dispatch offered by Julia.

GPU friendlyness will also require that your code does not use _scalar indexing_ (see the CUDA.jl documentation for more information), make sure to test your algorithm on the GPU while disallowing scalar indexing using `CUDA.allowscalar(false)`.

Finally, it is a good idea to implement the `Flux.gpu(yourpolicy)` and `cpu(yourpolicy)` functions, for user convenience. Be careful that sampling on the GPU requires a specific type of rng, you can generate one with `CUDA.default_rng()`
