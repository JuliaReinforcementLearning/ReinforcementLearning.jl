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
    Q = learner.approximator
    optimiser_state = learner.approximator.optimiser_state

    γ = learner.γ
    loss_func = learner.loss_func
    
    a = CartesianIndex.(batch.action, 1:length(batch.action))
    s, s′, nothing, r, t = batch
    s, s′, r, t = Flux.gpu(s), Flux.gpu(s′), Flux.gpu(r), Flux.gpu(t)

    # TODO: This can probably be made generic for all approximators??
    # TODO: withgradient is unnecessary, gradient should be sufficient
    # TODO: Look into batch / data loading / train! function
    grads = Flux.gradient(Q) do m
        # Evaluate model and loss inside gradient context:
        q = RLCore.forward(Q, s)[a]
        q′ = maximum(RLCore.forward(Q, s′); dims=1) |> vec
        G = @. r + γ * (1 - t) * q′
        loss = loss_func(G, q)
        Flux.ignore_derivatives() do
            learner.loss = loss
        end
        loss
    end
    # Optimization step
    Flux.update!(optimiser_state, Q.model, grads[1])
end
