export BasicDQNLearner

"""
    BasicDQNLearner(;kwargs...)

See paper: [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

This is the very basic implementation of DQN. Compared to the traditional Q learning, the only difference is that,
in the updating step it uses a batch of transitions sampled from an experience buffer instead of current transition.
And the `approximator` is usually a [`NeuralNetworkApproximator`](@ref).
You can start from this implementation to understand how everything is organized and how to write your own customized algorithm.

# Keywords

- `approximator`::[`AbstractApproximator`](@ref): used to get Q-values of a state.
- `loss_func`: the loss function to use.
- `γ::Float32=0.99f0`: discount rate.
- `batch_size::Int=32`
- `min_replay_history::Int=32`: number of transitions that should be experienced before updating the `approximator`.
- `rng=Random.GLOBAL_RNG`
"""
mutable struct BasicDQNLearner{Q,F,R} <: AbstractLearner
    approximator::Q
    loss_func::F
    γ::Float32
    sampler::BatchSampler
    min_replay_history::Int
    rng::R
    # for debugging
    loss::Float32
end

Flux.functor(x::BasicDQNLearner) = (Q = x.approximator,), y -> begin
    x = @set x.approximator = y.Q
    x
end

(learner::BasicDQNLearner)(env) =
    env |>
    state |>
    x -> send_to_device(device(learner), x) |> learner.approximator |> send_to_host

function BasicDQNLearner(;
    approximator::Q,
    loss_func::F = huber_loss,
    γ = 0.99f0,
    batch_size = 32,
    min_replay_history = 32,
    rng = Random.GLOBAL_RNG,
) where {Q,F}
    BasicDQNLearner{Q,F,typeof(rng)}(
        approximator,
        loss_func,
        γ,
        BatchSampler{SARTS}(batch_size),
        min_replay_history,
        rng,
        0.0,
    )
end

function RLBase.update!(learner::BasicDQNLearner, traj::AbstractTrajectory)
    if length(traj) >= learner.min_replay_history
        inds, batch = sample(learner.rng, traj, learner.sampler)
        update!(learner, batch)
    end
end

function RLBase.update!(learner::BasicDQNLearner, batch::NamedTuple{SARTS})

    Q = learner.approximator
    γ = learner.γ
    loss_func = learner.loss_func

    s, a, r, t, s′ = send_to_device(device(Q), batch)
    a = CartesianIndex.(a, 1:length(a))

    gs = gradient(params(Q)) do
        q = Q(s)[a]
        q′ = vec(maximum(Q(s′); dims = 1))
        G = @. r + γ * (1 - t) * q′
        loss = loss_func(G, q)
        ignore() do
            learner.loss = loss
        end
        loss
    end

    update!(Q, gs)
end
