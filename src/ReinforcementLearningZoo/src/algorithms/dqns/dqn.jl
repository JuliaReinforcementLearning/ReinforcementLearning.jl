export DQNLearner

using Random
using Flux

mutable struct DQNLearner{
    Tq<:AbstractApproximator,
    Tt<:AbstractApproximator,
    Tf,
    S<:Union{Int,Nothing},
    R<:AbstractRNG,
} <: AbstractLearner
    approximator::Tq
    target_approximator::Tt
    loss_func::Tf
    stack_size::S
    γ::Float32
    batch_size::Int
    update_horizon::Int
    min_replay_history::Int
    update_freq::Int
    target_update_freq::Int
    update_step::Int
    rng::R
end

function DQNLearner(;
    approximator::Tq,
    target_approximator::Tt,
    loss_func::Tf,
    stack_size::Union{Int,Nothing} = 4,
    γ::Float32 = 0.99f0,
    batch_size::Int = 32,
    update_horizon::Int = 1,
    min_replay_history::Int = 32,
    update_freq::Int = 1,
    target_update_freq::Int = 100,
    update_step::Int = 0,
    seed = nothing,
) where {Tq,Tt,Tf}
    copyto!(approximator, target_approximator)
    rng = MersenneTwister(seed)
    DQNLearner(
        approximator,
        target_approximator,
        loss_func,
        stack_size,
        γ,
        batch_size,
        update_horizon,
        min_replay_history,
        update_freq,
        target_update_freq,
        update_step,
        rng,
    )
end

"""

!!! note
    The state of the observation is assumed to have been stacked,
    if `!isnothing(stack_size)`.
"""
(learner::DQNLearner)(obs) =
    obs |> get_state |>
    x ->
        send_to_device(device(learner.approximator), x) |> learner.approximator |>
        send_to_host

function RLBase.update!(learner::DQNLearner, batch)
    learner.update_step += 1
    learner.update_step % learner.update_freq == 0 || return nothing

    Q, Qₜ, γ, loss_func, update_horizon, batch_size = learner.approximator,
    learner.target_approximator,
    learner.γ,
    learner.loss_func,
    learner.update_horizon,
    learner.batch_size
    states, rewards, terminals, next_states = map(
        x -> send_to_device(device(Q), x),
        (batch.states, batch.rewards, batch.terminals, batch.next_states),
    )
    actions = CartesianIndex.(batch.actions, 1:batch_size)

    gs = gradient(params(Q)) do
        q = batch_estimate(Q, states)[actions]
        q′ = dropdims(maximum(batch_estimate(Qₜ, next_states); dims = 1), dims = 1)
        G = rewards .+ γ^update_horizon .* (1 .- terminals) .* q′
        loss_func(G, q)
    end

    update!(Q, gs)

    if learner.update_step % learner.target_update_freq == 0
        copyto!(Qₜ, Q)
    end
end

function RLBase.extract_experience(t::AbstractTrajectory, learner::DQNLearner)
    s = learner.stack_size
    h = learner.update_horizon
    n = learner.batch_size
    γ = learner.γ
    valid_ind_range = isnothing(s) ? (1:(length(t)-h)) : (s:(1:(length(t)-h)))
    if length(t) > learner.min_replay_history
        inds = rand(learner.rng, valid_ind_range, n)
        states = consecutive_view(get_trace(t, :state), inds; n_stack = s)
        actions = consecutive_view(get_trace(t, :action), inds)
        next_states = consecutive_view(get_trace(t, :state), inds .+ h; n_stack = s)
        consecutive_rewards = consecutive_view(get_trace(t, :reward), inds; n_horizon = h)
        consecutive_terminals =
            consecutive_view(get_trace(t, :terminal), inds; n_horizon = h)
        rewards, terminals = zeros(Float32, n), fill(false, n)

        # make sure that we only consider experiences in current episode
        for i in 1:n
            m = findfirst(view(consecutive_terminals, :, i))
            if isnothing(m)
                terminals[i] = false
                rewards[i] = discount_rewards_reduced(view(consecutive_rewards, :, i), γ)
            else
                terminals[i] = true
                rewards[i] = discount_rewards_reduced(view(consecutive_rewards, 1:m, i), γ)
            end
        end
        (
            states = states,
            actions = actions,
            rewards = rewards,
            terminals = terminals,
            next_states = next_states,
        )
    else
        nothing
    end
end
