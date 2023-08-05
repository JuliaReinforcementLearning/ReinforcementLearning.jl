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

    γ = Flux.gpu(learner.γ)
    loss_func = learner.loss_func
    
    s, s′, a, r, t = Flux.gpu(batch)
    a = CartesianIndex.(a, 1:length(a))

    # TODO: This can probably be made generic for all approximators??
    # TODO: withgradient is unnecessary, gradient should be sufficient
    # TODO: Look into batch / data loading / train! function
    grads = Flux.gradient(Q) do m
        # Evaluate model and loss inside gradient context:
        q = RLCore.forward(Q, s)[a]
        q′ = maximum(RLCore.forward(Q, s′); dims=1) |> vec # TODO: pass to GPU
        G = @. r + γ * (1 - t) * q′
        loss = loss_func(G, q)
        ignore_derivatives() do
            learner.loss = loss
        end
        loss
    end
    # Optimization step
    Flux.update!(optimiser_state, Q.model, grads[1])
end


# using Flux
# using Metal

# G = Flux.gpu(Float32[1.0373293, 1.0380119, 0.99908245, 1.045342, 0.0, 1.0146753, 0.97506154, 0.99032134, 1.0777743, 0.9605096, 1.178752, 0.99887836, 1.1104985, 1.1282829, 1.0734904, 1.1108793, 0.99887836, 1.0777743, 0.9389985, 1.0408719, 0.99923605, 1.0789776, 1.0075705, 1.0853015, 1.0109681, 0.99130446, 1.0669373, 1.0852368, 0.9389985, 1.058615, 1.0181023, 1.0757337])

# q = Flux.gpu(Float32[-0.11436375, 0.011078824, -0.0020409964, 0.018125542, 0.010247405, 0.019373583, -0.07201143, -0.041619577, 0.049312495, -0.0499388, 0.14547554, -0.000926844, 0.057513118, 0.09047526, 0.045799907, 0.073412046, -0.000926844, 0.049312495, -0.039889317, 0.03770643, -0.00071363687, 0.016893676, -0.011023442, 0.058076072, 0.007646955, -0.041903377, -0.014816721, 0.0447669, -0.039889317, 0.038869392, -0.002664225, -0.013921689])

# gradient(Flux.huber_loss, G, q)

# using Statistics: mean, std
# ŷ, y= G, q
# abs_error = abs.(ŷ .- y)
# δ = 1
# temp = Zygote.ignore_derivatives(abs_error .<  δ)
# x = Flux.ofeltype(ŷ, 0.5)
# mean(((abs_error .^ 2) .* temp) .* x .+ δ * (abs_error .- x * δ) .* (1 .- temp))
# Flux.huber_loss(G, q, delta=1f0)
