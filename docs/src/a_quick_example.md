# A Quick Example

Welcome to the world of reinforcement learning in Julia! Here's a quick example to show you how to train an agent with a [`BasicDQNLearner`](@ref) to play the `CartPoleEnv`.

!!! note
    Notice that a lot of dependent packages are under rapid development. To make sure that you can reproduce the result in this example, you are suggested to:
    1. Make sure that your Julia version is `v1.3-rc3` or above
    1. Clone `git@github.com:JuliaReinforcementLearning/ReinforcementLearning.jl.git`
    1. `cd ReinforcementLearning.jl`
    1. `julia --project=docs`
    1. `]instantiate`

First, let's make sure that running the following code will not trigger any error:

```@example 1
import Random # hide
Random.seed!(1) # hide

using ReinforcementLearning, ReinforcementLearningEnvironments, Flux
using StatsBase:mean
```

Cartpole is considered to be one of the simplest environments for **DRL(Deep Reinforcement Learning)** algorithms testing. The state of the Cartpole environment can be described with 4 numbers and the actions are two integers(`1` and `2`). Before game terminates, agent can gain a reward of `+1` for each step. And the game will be forced to end after 200 steps, thus the maximum reward of an episode is **200**. 

```@example 1
env = CartPoleEnv(;T=Float32)
ns, na = length(observation_space(env)), length(action_space(env))  # (4, 2)
```

Then we can create an agent:

```@example 1
backend  = :Zygote
device = :cpu

agent = Agent(
    π = QBasedPolicy(
        learner = BasicDQNLearner(
            approximator = NeuralNetworkQ(
                model = Chain(
                    Dense(ns, 128, relu; backend=backend),
                    Dense(128, 128, relu; backend=backend),
                    Dense(128, na; backend=backend)
                    ),
                optimizer = ADAM(),
                device = device
            ),
            batch_size = 32,
            min_replay_history = 100,
            loss_fun = huber_loss,
        ),
        selector = EpsilonGreedySelector{:exp}(ϵ_stable = 0.01, decay_steps = 500),
    ),
    buffer = circular_RTSA_buffer(
        capacity = 1000,
        state_eltype = Float32,
        state_size = (ns,),
    )
)
```

Relax! We promise that all the new concepts above will be explained in detail later.

For now, you only need to know that an [`Agent`](@ref) is usually composed by a *policy* and a *buffer*. Here we are using a very common [`QBasedPolicy`](@ref) and a [`circular_RTSA_buffer`](@ref). For a [`QBasedPolicy`](@ref) we need to provide a *learner* and a *selector*. The *learner* here is used to provide the value estimations of all actions in a step, and the *selector* is use to select an action based on those estimations. For a *buffer*, it stores some transitions between an *agent* and an *environment* and is used to improve the *policy*. That's all!

To record the reward and performance , we need some hooks:

```@example 1
hook = ComposedHook(
    TotalRewardPerEpisode(),
    TimePerStep()
)
```

And finally, let's push the button:

```@example 1
run(agent, env, StopAfterStep(10000; is_show_progress=false); hook = hook)

print("""
    backend = $backend, device = $device
    avg_reward = $(mean(hook[1].rewards))
    avg_fps = $(1/mean(hook[2].times))
    """)
```

We can also plot the rewards stored in our hook:

```@example 1
using Plots
plot(hook[1].rewards, xlabel="Episode", ylabel="Reward", label="")
savefig("a_quick_example_cartpole_cpu_basic_dqn.png"); nothing # hide
```

![](a_quick_example_cartpole_cpu_basic_dqn.png)


**That's fantastic!**

- *"But I'm new to Julia and RL. Can I learn RL by using this package?"*

    Yes! One of this package's main goals is to be educational. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) is a good introductory book. And we reproduce almost all the examples mentioned in that book by using this package [here](https://github.com/JuliaReinforcementLearning/ReinforcementLearningAnIntroduction.jl).

- *"What if I have a solid background in RL but new to Julia?"*

    > Programming isn't hard. Programming **well** is **very** hard!  - [CS 3110](https://www.cs.cornell.edu/courses/cs3110/)

    Fortunately, Julia provides some amazing features together with many awesome packages to make things much easier. We provide a [Tips for Developers](@ref) section to help you grasp Julia in depth.

- *"I'm experienced in both Julia and RL. But I find it hard to use this package..."*

    Although we tried our best to make concepts and codes as simple as possible, it is still possible that they are not very intuitive enough. So do not hesitate to **JOIN US** (create an issue or a PR). We need **YOU** to improve all this stuff together!