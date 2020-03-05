export PrioritizedDQNLearner

using Random
using Flux
using Zygote
using StatsBase: mean

"""
    PrioritizedDQNLearner(;kwargs...)

See paper: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

# Keywords

- `approximator`::[`AbstractApproximator`](@ref): used to get Q-values of a state.
- `target_approximator`::[`AbstractApproximator`](@ref): similar to `approximator`, but used to estimate the target (the next state).
- `loss_func`: the loss function.
- `γ::Float32=0.99f0`: discount rate.
- `batch_size::Int=32`
- `update_horizon::Int=1`: length of update ('n' in n-step update).
- `min_replay_history::Int=32`: number of transitions that should be experienced before updating the `approximator`.
- `update_freq::Int=4`: the frequency of updating the `approximator`.
- `target_update_freq::Int=100`: the frequency of syncing `target_approximator`.
- `stack_size::Union{Int, Nothing}=4`: use the recent `stack_size` frames to form a stacked state.
- `default_priority::Float64=100.`: the default priority for newly added transitions.
- `seed = nothing`
"""
mutable struct PrioritizedDQNLearner{
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
    default_priority::Float32
    rng::R
end

function PrioritizedDQNLearner(;
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
    default_priority::Float32 = 100f0,
    seed = nothing,
) where {Tq,Tt,Tf}
    copyto!(approximator, target_approximator)
    rng = MersenneTwister(seed)
    PrioritizedDQNLearner(
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
        default_priority,
        rng,
    )
end

"""

!!! note
    The state of the observation is assumed to have been stacked,
    if `!isnothing(stack_size)`.
"""
(learner::PrioritizedDQNLearner)(obs) =
    obs |> get_state |>
    x ->
        send_to_device(device(learner.approximator), x) |> learner.approximator |>
        send_to_host

function RLBase.update!(learner::PrioritizedDQNLearner, batch)
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

    priorities = send_to_device(device(Q), Vector{Float32}())

    gs = gradient(params(Q)) do
        q = batch_estimate(Q, states)[actions]
        q′ = dropdims(maximum(batch_estimate(Qₜ, next_states); dims = 1), dims = 1)
        G = rewards .+ γ^update_horizon .* (1 .- terminals) .* q′

        batch_losses = loss_func(G, q)
        priorities = (Zygote.dropgrad(batch_losses) .+ 1f-10)
        mean(batch_losses)
    end

    update!(Q, gs)

    if learner.update_step % learner.target_update_freq == 0
        copyto!(Qₜ, Q)
    end

    send_to_host(priorities)
end

function RLBase.extract_experience(t::AbstractTrajectory, learner::PrioritizedDQNLearner)
    s = learner.stack_size
    h = learner.update_horizon
    n = learner.batch_size
    γ = learner.γ
    if length(t) > learner.min_replay_history
        # 1. sample indices based on priority
        inds = Vector{Int}(undef, n)
        valid_ind_range = isnothing(s) ? (1:(length(t)-h)) : (s:(1:(length(t)-h)))
        for i in 1:n
            ind, p = sample(learner.rng, get_trace(t, :priority))
            while ind ∉ valid_ind_range
                ind, p = sample(learner.rng, get_trace(t, :priority))
            end
            inds[i] = ind
        end

        # 2. extract SARTS
        states = consecutive_view(get_trace(t, :state), inds; n_stack = s)
        actions = consecutive_view(get_trace(t, :action), inds)
        next_states = consecutive_view(get_trace(t, :state), inds .+ h; n_stack = s)
        consecutive_rewards = consecutive_view(get_trace(t, :reward), inds; n_horizon = h)
        consecutive_terminals =
            consecutive_view(get_trace(t, :terminal), inds; n_horizon = h)
        rewards, terminals = zeros(Float32, n), fill(false, n)

        rewards = discount_rewards_reduced(
            consecutive_rewards,
            γ;
            terminal = consecutive_terminals,
            dims = 1,
        )
        terminals = mapslices(any, consecutive_terminals; dims = 1) |> vec

        inds,
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

function RLBase.update!(p::QBasedPolicy{<:PrioritizedDQNLearner}, t::AbstractTrajectory)
    indexed_experience = extract_experience(t, p)
    if !isnothing(indexed_experience)
        inds, experience = indexed_experience
        priorities = update!(p.learner, experience)
        if !isnothing(priorities)
            get_trace(t, :priority)[inds] .= priorities
        end
    end
end

function (
    agent::Agent{
        <:QBasedPolicy{<:PrioritizedDQNLearner},
        <:CircularCompactPSARTSATrajectory,
    }
)(
    ::PostActStage,
    obs,
)
    push!(
        agent.trajectory;
        reward = get_reward(obs),
        terminal = get_terminal(obs),
        priority = agent.policy.learner.default_priority,
    )
    nothing
end
