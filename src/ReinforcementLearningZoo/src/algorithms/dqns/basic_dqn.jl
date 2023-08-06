export BasicDQNLearner

using Flux
using Flux: gradient
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
Base.@kwdef mutable struct BasicDQNLearner{Q, F} <: AbstractLearner
    approximator::Q
    loss_func::F = huber_loss
    γ::Float32 = 0.99f0
    # for debugging
    loss::Float32 = 0.0f0
end

@functor BasicDQNLearner (approximator,)

RLCore.forward(L::BasicDQNLearner, s::A) where {A<:AbstractArray} = RLCore.forward(L.approximator, s)

function RLCore.optimise!(learner::BasicDQNLearner, ::PostActStage, trajectory::Trajectory)
    for batch in trajectory
        optimise!(learner, batch)
    end
end

function RLCore.optimise!(
    learner::BasicDQNLearner,
    batch::NamedTuple
)
    approx = learner.approximator
    optimiser_state = learner.approximator.optimiser_state

    γ = learner.γ
    loss_func = learner.loss_func

    s, s′, a, r, t = Flux.gpu(batch)
    a = CartesianIndex.(a, 1:length(a))

    model = approx.model

    grads = Flux.gradient(model) do Q
        # Evaluate model and loss inside gradient context:
        q = Q(s)[a]
        q′ = maximum(Q(s′); dims=1) |> vec
        G = @. r + γ * (1 - t) * q′
        loss_ = loss_func(G, q)
        Flux.ignore_derivatives() do
            learner.loss = loss_
        end
        loss_
    end |> Flux.cpu

    # Optimization step
    Flux.update!(optimiser_state, Flux.cpu(model), grads[1])
end
