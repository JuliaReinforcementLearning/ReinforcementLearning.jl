export BasicDQNLearner

using Flux
using Flux: gradient, params
using Functors: @functor
using ChainRulesCore: ignore_derivatives

"""
    BasicDQNLearner(;kwargs...)

See paper: [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

This is the very basic implementation of DQN. Compared to the traditional Q
learning, the only difference is that, in the optimising step it uses a batch of
transitions sampled from an experience buffer instead of current transition. And
a neural network is used to estimate the Q-value.  You can start from this
implementation to understand how everything is organized and how to write your
own customized algorithm.

# Keyword Arguments

- `approximator`::[`Approximator`](@ref): used to get Q-values of a state.
- `loss_func=huber_loss`: the loss function to use.
- `γ::Float32=0.99f0`: discount rate.
"""
Base.@kwdef mutable struct BasicDQNLearner{Q} <: AbstractLearner
    approximator::Q
    loss_func::Any = huber_loss
    γ::Float32 = 0.99f0
    # for debugging
    loss::Float32 = 0.0f0
end

@functor BasicDQNLearner (approximator,)

RLBase.optimise!(L::BasicDQNLearner, s::AbstractArray) = RLBase.optimise!(L.approximator, s)

function RLCore.optimise!(
    learner::BasicDQNLearner,
    batch::NamedTuple{SS′ART},
)

    Q = learner.approximator
    γ = learner.γ
    loss_func = learner.loss_func

    s, s′, a, r, t = send_to_device(device(Q), batch)
    a = CartesianIndex.(a, 1:length(a))

    gs = gradient(params(Q)) do
        q = RLCore.estimate_reward(Q, s)[a]
        q′ = vec(maximum(RLCore.estimate_reward(Q, s′); dims=1))
        G = @. r + γ * (1 - t) * q′
        loss = loss_func(G, q)
        ignore_derivatives() do
            learner.loss = loss
        end
        loss
    end

    optimise!(Q, gs)
end
