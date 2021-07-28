# How to use hooks?

## What are the hooks?

During the interactions between agents and environments, we often want to
collect some useful information. One straightforward approach is the imperative
programming. We write the code in a loop and execute them step by step.

```julia
while true
    env |> policy |> env
    # write your own logic here
    # like saving parameters, recording loss function, evaluating policy, etc.
    stop_condition(env, policy) && break
    is_terminated(env) && reset!(env)
end
```

The benifit of this approach is the great clarity. You are responsible for what
you write. And this is the encouraged approach for new users to try different
components in this package.

Another approach is the declarative programming. We describe when and what we
want to do during an experiment. Then put them together with the agent and
environment. Finally we execute the `run` command to conduct our experiment. In
this way, we can reuse some common hooks and execution pipelines instead of
writing many duplicate codes. In many existing reinforcement learning python
packages, people usually use a set of configuration files to define the
execution pipeline. However, we believe this is not necessary in Julia. With the
declarative programming approach, we gain much more flexibilities.

Now the question is how to design the hook. A natural choice is to wrap the
comments part in the above pseudocode into a function:

```julia
while true
    env |> policy |> env
    hook(policy, env)
    stop_condition(env, policy) && break
    is_terminated(env) && reset!(env)
end
```

But sometimes, we'd like to have a more fingrained control. So we split the calling
of hooks into several different stages:

- [`PreExperimentStage`](@ref)
- [`PreEpisodeStage`](@ref)
- [`PreActStage`](@ref)
- [`PostActStage`](@ref)
- [`PostEpisodeStage`](@ref)
- [`PostExperimentStage`](@ref)

## How to define a customized hook?

By default, an instance of [`AbstractHook`](@ref) will do nothing when called
with `(hook::AbstractHook)(::AbstractStage, policy, env)`. So when writing a
customized hook, you only need to implement the necessary runtime logic.

For example, assume we want to record the wall time of each episode.

```@repl how_to_use_hooks
using ReinforcementLearning
Base.@kwdef mutable struct TimeCostPerEpisode <: AbstractHook
    t::UInt64 = time_ns()
    time_costs::Vector{UInt64} = []
end
(h::TimeCostPerEpisode)(::PreEpisodeStage, policy, env) = h.t = time_ns()
(h::TimeCostPerEpisode)(::PostEpisodeStage, policy, env) = push!(h.time_costs, time_ns()-h.t)
h = TimeCostPerEpisode()
run(RandomPolicy(), CartPoleEnv(), StopAfterEpisode(10), h)
h.time_costs
```

## Common hooks

- [`StepsPerEpisode`](@ref)
- [`RewardsPerEpisode`](@ref)
- [`TotalRewardPerEpisode`](@ref)
- [`TotalBatchRewardPerEpisode`](@ref)

## Periodic jobs

Sometimes, we'd like to periodically run some functions. Two handy hooks are
provided for this kind of tasks:

- [`DoEveryNEpisode`](@ref)
- [`DoEveryNStep`](@ref)

Following are some typical usages.

### Evaluating policy during training

```@repl how_to_use_hooks
using Statistics: mean
policy = RandomPolicy()
run(
    policy,
    CartPoleEnv(),
    StopAfterEpisode(100),
    DoEveryNEpisode(;n=10) do t, policy, env
        # In real world cases, the policy is usually wrapped in an Agent,
        # we need to extract the inner policy to run it in the *actor* mode.
        # Here for illustration only, we simply use the origina policy.

        # Note that we create a new instance of CartPoleEnv here to avoid
        # polluting the original env.

        hook = TotalRewardPerEpisode(;is_display_on_exit=false)
        run(policy, CartPoleEnv(), StopAfterEpisode(10), hook)

        # now you can report the result of the hook.
        println("avg reward at episode $t is: $(mean(hook.rewards))")
    end
)
```

### Save parameters

[BSON.jl](https://github.com/JuliaIO/BSON.jl) is recommended to save the parameters of a policy.

```@repl how_to_use_hooks
using Flux
using Flux.Losses: huber_loss
using BSON

env = CartPoleEnv(; T = Float32)
ns, na = length(state(env)), length(action_space(env))

policy = Agent(
    policy = QBasedPolicy(
        learner = BasicDQNLearner(
            approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 128, relu; init = glorot_uniform),
                    Dense(128, 128, relu; init = glorot_uniform),
                    Dense(128, na; init = glorot_uniform),
                ) |> cpu,
                optimizer = ADAM(),
            ),
            batch_size = 32,
            min_replay_history = 100,
            loss_func = huber_loss,
        ),
        explorer = EpsilonGreedyExplorer(
            kind = :exp,
            ϵ_stable = 0.01,
            decay_steps = 500,
        ),
    ),
    trajectory = CircularArraySARTTrajectory(
        capacity = 1000,
        state = Vector{Float32} => (ns,),
    ),
)

parameters_dir = mktempdir()

run(
    policy,
    env,
    StopAfterStep(10_000),
    DoEveryNStep(n=1_000) do t, p, e
        ps = params(p)
        f = joinpath(parameters_dir, "parameters_at_step_$t.bson")
        BSON.@save f ps
        println("parameters at step $t saved to $f")
    end
)
```

### Logging data

Below we demonstrate how to use
[TensorBoardLogger.jl](https://github.com/PhilipVinc/TensorBoardLogger.jl) to
log runtime metrics. But users could also other tools like
[wandb](https://wandb.ai/site) through
[PyCall.jl](https://github.com/JuliaPy/PyCall.jl).


```@repl how_to_use_hooks
using TensorBoardLogger
using Logging
tf_log_dir = "logs"
lg = TBLogger(tf_log_dir, min_level = Logging.Info)
total_reward_per_episode = TotalRewardPerEpisode()
hook = ComposedHook(
    total_reward_per_episode,
    DoEveryNEpisode() do t, agent, env
        with_logger(lg) do
            @info "training"  reward = total_reward_per_episode.rewards[end]
        end
    end
)
run(RandomPolicy(), CartPoleEnv(), StopAfterEpisode(50), hook)
readdir(tf_log_dir)
```

Then run `tensorboard --logdir logs` and open the link on the screen in your
browser. (Obviously you need to install tensorboard first.)