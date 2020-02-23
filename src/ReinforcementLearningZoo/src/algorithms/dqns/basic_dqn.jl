export BasicDQNLearner

using Random
using Flux

struct BasicDQNLearner{Q,F,R} <: AbstractLearner
    approximator::Q
    loss_func::F
    γ::Float32
    batch_size::Int
    min_replay_history::Int
    rng::R
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
    )
end

function RLBase.update!(learner::BasicDQNLearner, batch)
    Q, γ, loss_func, batch_size =
        learner.approximator, learner.γ, learner.loss_func, learner.batch_size
    s, r, t, s′ = map(
        x -> send_to_device(device(Q), x),
        (batch.state, batch.reward, batch.terminal, batch.next_state),
    )
    a = CartesianIndex.(batch.action, 1:batch_size)

    gs = gradient(params(Q)) do
        q = batch_estimate(Q, s)[a]
        q′ = vec(maximum(batch_estimate(Q, s′); dims = 1))
        G = r .+ γ .* (1 .- t) .* q′
        loss_func(G, q)
    end

    update!(Q, gs)
end

function RLBase.extract_experience(
    t::CircularCompactSARTSATrajectory,
    learner::BasicDQNLearner,
)
    if length(t) > learner.min_replay_history
        inds = rand(learner.rng, 1:length(t), learner.batch_size)
        map(get_trace(t, :state, :action, :reward, :terminal, :next_state)) do x
            consecutive_view(x, inds)
        end
    else
        nothing
    end
end
