export BasicDQNLearner

using Random
using Flux

"""
    BasicDQNLearner(;kwargs...)

See paper: [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

This is the very basic implementation of DQN. Compared to the traditional Q learning, the only difference is that,
in the updating step it uses a batch of transitions sampled from an experience buffer instead of current transition.
And the `approximator` is usually a [`NeuralNetworkApproximator`](@ref).
You can start from this implementation to understand how everything is organized and how to write your own customized algorithm.
# Keywords
- `approximator`::[`AbstractApproximator`](@ref): used to get Q-values of a state.
- `loss_func`: the loss function to use. TODO: provide a default [`huber_loss`](@ref)?
- `γ::Float32=0.99f0`: discount rate.
- `batch_size::Int=32`
- `min_replay_history::Int=32`: number of transitions that should be experienced before updating the `approximator`.
- `seed=nothing`.
"""
mutable struct BasicDQNLearner{Q,F,R} <: AbstractLearner
    approximator::Q
    loss_func::F
    γ::Float32
    batch_size::Int
    min_replay_history::Int
    rng::R
    loss::Float32
end

(learner::BasicDQNLearner)(obs) =
    obs |> get_state |>
    x ->
        send_to_device(device(learner.approximator), x) |> learner.approximator |>
        send_to_host

function BasicDQNLearner(;
    approximator::Q,
    loss_func::F,
    γ = 0.99f0,
    batch_size = 32,
    min_replay_history = 32,
    seed = nothing,
) where {Q,F}
    rng = MersenneTwister(seed)
    BasicDQNLearner{Q,F,typeof(rng)}(
        approximator,
        loss_func,
        γ,
        batch_size,
        min_replay_history,
        rng,
        0.
    )
end

function RLBase.update!(learner::BasicDQNLearner, t::AbstractTrajectory)
    length(t) < learner.min_replay_history && return

    inds = rand(learner.rng, 1:length(t), learner.batch_size)
    batch = map(get_trace(t, :state, :action, :reward, :terminal, :next_state)) do x
        consecutive_view(x, inds)
    end

    Q, γ, loss_func, batch_size =
        learner.approximator, learner.γ, learner.loss_func, learner.batch_size
    s, r, t, s′ = map(
        x -> send_to_device(device(Q), x),
        (batch.state, batch.reward, batch.terminal, batch.next_state),
    )
    a = CartesianIndex.(batch.action, 1:batch_size)

    gs = gradient(params(Q)) do
        q = Q(s)[a]
        q′ = vec(maximum(Q(s′); dims = 1))
        G = r .+ γ .* (1 .- t) .* q′
        loss = loss_func(G, q)
        ignore() do
            learner.loss = loss
        end
        loss
    end

    update!(Q, gs)
end
