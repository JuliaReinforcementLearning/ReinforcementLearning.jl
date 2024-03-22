# How to use hooks?

## What are the hooks?

During the interactions between agents and environments, we often want to
collect some useful information. One straightforward approach is the imperative
programming. We write the code in a loop and execute them step by step.

```julia
while true
    action = plan!(policy, env)
    act!(env, action)

    # write your own logic here
    # like saving parameters, recording loss function, evaluating policy, etc.
    check!(stop_condition, env, policy) && break
    is_terminated(env) && reset!(env)
end
```

The benefit of this approach is the great clarity. You are responsible for what
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
comments part in the above pseudo-code into a function:

```julia
while true
    action = plan!(policy, env)
    act!(env, action)
    push!(hook, policy, env)
    check!(stop_condition, env, policy) && break
    is_terminated(env) && reset!(env)
end
```

But sometimes, we'd like to have a more fine-grained control. So we split the calling
of hooks into several different stages:

- [`PreExperimentStage`](@ref)
- [`PreEpisodeStage`](@ref)
- [`PreActStage`](@ref)
- [`PostActStage`](@ref)
- [`PostEpisodeStage`](@ref)
- [`PostExperimentStage`](@ref)

## How to define a customized hook?

By default, an instance of [`AbstractHook`](@ref) will do nothing when called
with `push!(hook::AbstractHook, ::AbstractStage, policy, env)`. So when writing a
customized hook, you only need to implement the necessary runtime logic.

For example, assume we want to record the wall time of each episode.

```@repl how_to_use_hooks
using ReinforcementLearning
import Base.push!
Base.@kwdef mutable struct TimeCostPerEpisode <: AbstractHook
    t::UInt64 = time_ns()
    time_costs::Vector{UInt64} = []
end
Base.push!(h::TimeCostPerEpisode, ::PreEpisodeStage, policy, env) = h.t = time_ns()
Base.push!(h::TimeCostPerEpisode, ::PostEpisodeStage, policy, env) = push!(h.time_costs, time_ns()-h.t)
h = TimeCostPerEpisode()

run(RandomPolicy(), CartPoleEnv(), StopAfterNEpisodes(10), h)
h.time_costs
```

## Common hooks

- [`StepsPerEpisode`](@ref)
- [`RewardsPerEpisode`](@ref)
- [`TotalRewardPerEpisode`](@ref)

## Periodic jobs

Sometimes, we'd like to periodically run some functions. Two handy hooks are
provided for this kind of tasks:

- [`DoEveryNEpisodes`](@ref)
- [`DoEveryNSteps`](@ref)

Following are some typical usages.

### Evaluating policy during training

```@repl how_to_use_hooks
using Statistics: mean
policy = RandomPolicy()
run(
    policy,
    CartPoleEnv(),
    StopAfterNEpisodes(100),
    DoEveryNEpisodes(;n=10) do t, policy, env
        # In real world cases, the policy is usually wrapped in an Agent,
        # we need to extract the inner policy to run it in the *actor* mode.
        # Here for illustration only, we simply use the original policy.

        # Note that we create a new instance of CartPoleEnv here to avoid
        # polluting the original env.

        hook = TotalRewardPerEpisode(;is_display_on_exit=false)
        run(policy, CartPoleEnv(), StopAfterNEpisodes(10), hook)

        # now you can report the result of the hook.
        println("avg reward at episode $t is: $(mean(hook.rewards))")
    end
)
```

### Save parameters

[BSON.jl](https://github.com/JuliaIO/BSON.jl) is recommended to save the parameters of a policy.

```@repl how_to_use_hooks
using ReinforcementLearning
using BSON

env = RandomWalk1D()
ns, na = length(state_space(env)), length(action_space(env))

policy = Agent(
    QBasedPolicy(;
        learner = TDLearner(
            TabularQApproximator(n_state = ns, n_action = na),
            :SARS;
        ),
        explorer = EpsilonGreedyExplorer(ϵ_stable=0.01),
    ),
    Trajectory(
        CircularArraySARTSTraces(;
            capacity = 1,
            state = Int64 => (),
            action = Int64 => (),
            reward = Float64 => (),
            terminal = Bool => (),
        ),
        DummySampler(),
        InsertSampleRatioController(),
    ),
)

parameters_dir = mktempdir()

run(
    policy,
    env,
    StopAfterNSteps(10_000),
    DoEveryNSteps(n=1_000) do t, p, e
        ps = policy.policy.learner.approximator
        f = joinpath(parameters_dir, "parameters_at_step_$t.bson")
        BSON.@save f ps
        println("parameters at step $t saved to $f")
    end
)
```
