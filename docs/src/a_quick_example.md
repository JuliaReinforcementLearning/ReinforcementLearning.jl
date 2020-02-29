# A Quick Example

Welcome to the world of reinforcement learning in Julia! Here's a quick example to show you how to train an agent with to play the [`CartPoleEnv`](@ref).

First, let's make sure that this package is properly installed.

```
using ReinforcementLearning
```

Cartpole is considered to be one of the simplest environments for **DRL(Deep Reinforcement Learning)** algorithms testing. The state of the Cartpole environment can be described with 4 numbers and the actions are two integers(`1` and `2`). Before game terminates, agent can gain a reward of `+1` for each step. And the game will be forced to end after 200 steps, thus the maximum reward of an episode is **200**. 

```
env = CartPoleEnv(;T=Float32, seed=123)
```

Then we create an agent to play with the cartpole environment.

```
agent = Agent(
    policy = RandomPolicy(env;seed=456),
    trajectory = CircularCompactSARTSATrajectory(; capacity=3, state_type=Float32, state_size = (4,)),
)
```

An agent is usually constructed by a policy and a trajectory. A policy is a mapping from an observation to an action. And a trajectory is used to store some important information of the interactions between agents and environments. The [`RandomPolicy`](@ref) used here will do nothing but select an action randomly. And the [`CircularCompactSARTSATrajectory`](@ref) here will store the **S**tate, **A**ction, **R**eward, **T**erminal, next-**S**tate and next-**A**ction in each step of the latest 3 episodes.

Now we can start to run simulations:

```
run(agent, env, StopAfterEpisode(1))
```

Here the [`StopAfterEpisode`](@ref)`(1)` is a stop condition, which means stop after `1` episode here.
Then we can take a look at the trajectory in the agent.

```
agent.trajectory
# 3-element Trajectory{(:state, :action, :reward, :terminal, :next_state, :next_action),Tuple{Float32,Int64,Float32,Bool,Float32,Int64},NamedTuple{(:reward, :terminal, :state, :action),Tuple{CircularArrayBuffer{Float32,1},CircularArrayBuffer{Bool,1},CircularArrayBuffer{Float32,2},CircularArrayBuffer{Int64,1}}}}:
#  (state = Float32[-0.116456345, -0.57231975, 0.16624497, 1.1284109], action = 2, reward = 1.0f0, terminal = false, next_state = Float32[-0.12790275, -0.37971866, 0.18881318, 0.89214355], next_action = 2)
#  (state = Float32[-0.12790275, -0.37971866, 0.18881318, 0.89214355], action = 2, reward = 1.0f0, terminal = false, next_state = Float32[-0.13549712, -0.18759018, 0.20665605, 0.6642545], next_action = 1) 
#  (state = Float32[-0.13549712, -0.18759018, 0.20665605, 0.6642545], action = 1, reward = 0.0f0, terminal = true, next_state = Float32[-0.13924892, -0.38489604, 0.21994114, 1.0142413], next_action = 2)
```

!!! note
    Since we have set the random seed of the [`RandomPolicy`](@ref) and the [`CartPoleEnv`](@ref), you should see the exactly same result as above.

To record the total reward of each episode, we can add a hook during each `run`.

```
hook = TotalRewardPerEpisode()
run(agent, env, StopAfterEpisode(10000), hook)
sum(hook.rewards)/10000  # 21.0591
```

Playing the cartpole environment with a [`RandomPolicy`](@ref) is not very interesting. Next we will use a DQN to solve the problem.

Because this package relies on [Flux.jl](https://github.com/FluxML/Flux.jl) to build deep learning models, you need to manually install Flux to run the following example. To show the rewards of each episode and the fps, [Plots.jl](https://github.com/JuliaPlots/Plots.jl) and [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl) are also required.

```
using Flux
using ReinforcementLearning
using StatsBase:mean

env = CartPoleEnv(; T = Float32, seed = 11)
ns, na = length(rand(get_observation_space(env))), length(get_action_space(env))
agent = Agent(
    policy = QBasedPolicy(
        learner = BasicDQNLearner(
            approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 128, relu; initW = seed_glorot_uniform(seed = 17)),
                    Dense(128, 128, relu; initW = seed_glorot_uniform(seed = 23)),
                    Dense(128, na; initW = seed_glorot_uniform(seed = 39)),
                ) |> gpu,
                optimizer = ADAM(),
            ),
            batch_size = 32,
            min_replay_history = 100,
            loss_func = huber_loss,
            seed = 22,
        ),
        explorer = EpsilonGreedyExplorer(
            kind = :exp,
            ϵ_stable = 0.01,
            decay_steps = 500,
            seed = 33,
        ),
    ),
    trajectory = CircularCompactSARTSATrajectory(
        capacity = 1000,
        state_type = Float32,
        state_size = (ns,),
    ),
)
hook = ComposedHook(TotalRewardPerEpisode(), TimePerStep())
run(agent, env, StopAfterStep(10000), hook)

@info "stats for BasicDQNLearner" avg_reward = mean(hook[1].rewards) avg_fps = 1 / mean(hook[2].times)
# ┌ Info: stats for BasicDQNLearner
# │   avg_reward = 107.43478260869566
# └   avg_fps = 531.283841452491
```

The main difference here is that, now we are using a [`QBasedPolicy`](@ref) instead of a [`RandomPolicy`](@ref). A model of three Dense layers is used to calculate the Q values.

Relax! We promise that all the new concepts above will be explained in detail later.

Now we can also plot the rewards stored in our hook:

```
using Plots
plot(hook[1].rewards, xlabel="Episode", ylabel="Reward", label="")
```

![](/assets/img/a_quick_example_cartpole_cpu_basic_dqn.png)


**That's fantastic!**

- *"But I'm new to Julia and RL. Can I learn RL by using this package?"*

    Yes! One of this package's main goals is to be educational. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) is a good introductory book. And we reproduce almost all the examples mentioned in that book by using this package [here](https://github.com/JuliaReinforcementLearning/ReinforcementLearningAnIntroduction.jl).

- *"What if I have a solid background in RL but new to Julia?"*

    > Programming isn't hard. Programming **well** is **very** hard!  - [CS 3110](https://www.cs.cornell.edu/courses/cs3110/)

    Fortunately, Julia provides some amazing features together with many awesome packages to make things much easier. We provide a [Tips for Developers](@ref) section to help you grasp Julia in depth.

- *"I'm experienced in both Julia and RL. But I find it hard to use this package..."*

    Although we tried our best to make concepts and codes as simple as possible, it is still possible that they are not very intuitive enough. So do not hesitate to **JOIN US** (create an issue or a PR). We need **YOU** to improve all this stuff together!