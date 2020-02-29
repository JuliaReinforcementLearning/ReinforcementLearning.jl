# Overview

Before diving into details, let's review some basic concepts in **RL(Reinforcement Learning)** first. Then we'll gradually introduce the relationship between those concepts and our implementations in this package.

## Key Concepts

### Agent and Environment

```@raw html
<img src="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/raw/master/docs/src/assets/img/agent_env_relation.png" width="640px">
```

Generally speaking, RL is to learn how to take actions so as to maximize a numerical reward. Two core concepts in RL are **Agent** and **Environment**. In each step, the agent is provided with the observation of the environment and is required to take an action. Then the environment consumes that action and transites to another state, providing a numerical reward in the meantime.

!!! note

    Sometimes people are confused about the concept of **state** and **observation** (see also the discussion [here](https://ai.stackexchange.com/questions/5970/what-is-the-difference-between-an-observation-and-a-state-in-reinforcement-learn)). In this package, we treat all the information we can get from the perspective of an agent in each step as an **observation**, including *state*, *reward* and some other extra *info*.

    Here we adopt the idea of [Duck typing](https://en.wikipedia.org/wiki/Duck_typing) to describe the observation from an environment. See [Environments](@ref) for more details.

In this package, [**Agent**](@ref) is one of the most typical subtypes of [`AbstractAgent`](@ref). And you may find different kinds of **Environment**s provided in [ReinforcementLearningEnvironments.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl). Agents and environments are required to be functional objects. So we can use the piping operator (`|>`) to simulate the steps implied in the above picture: `env |> observe |> agent |> env`. See [Agents](@ref) and [Environments](@ref) for more some concrete implementations.
