export DQNLearner

using Random
using Flux

"""
    DQNLearner(;kwargs...)
See paper: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
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
- `rng = Random.GLOBAL_RNG`
"""
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
    loss::Float32
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
    rng = Random.GLOBAL_RNG,
) where {Tq,Tt,Tf}
    copyto!(approximator, target_approximator)
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
        0.f0,
    )
end


Flux.functor(x::DQNLearner) = (Q = x.approximator, Qₜ = x.target_approximator),
y -> begin
    x = @set x.approximator = y.Q
    x = @set x.target_approximator = y.Qₜ
    x
end

"""

!!! note
    The state of the observation is assumed to have been stacked,
    if `!isnothing(stack_size)`.
"""
function (learner::DQNLearner)(env)
    probs = env |>
    get_state |>
    x -> Flux.unsqueeze(x, ndims(x) + 1) |>
    x -> send_to_device(device(learner.approximator), x) |>
    learner.approximator |>
    vec |>
    send_to_host

    if ActionStyle(env) === FULL_ACTION_SET
        probs .+= typemin(eltype(probs)) .* (1 .- get_legal_actions_mask(env))
    end
    probs
end

function RLBase.update!(learner::DQNLearner, t::AbstractTrajectory)
    length(t[:terminal]) < learner.min_replay_history && return

    learner.update_step += 1

    if learner.update_step % learner.target_update_freq == 0
        copyto!(learner.target_approximator, learner.approximator)
    end

    learner.update_step % learner.update_freq == 0 || return

    experience = extract_experience(t, learner)

    Q = learner.approximator
    Qₜ = learner.target_approximator
    γ = learner.γ
    loss_func = learner.loss_func
    update_horizon = learner.update_horizon
    batch_size = learner.batch_size
    D = device(Q)
    states = send_to_device(D, experience.states)
    actions = CartesianIndex.(experience.actions, 1:batch_size)
    rewards = send_to_device(D, experience.rewards)
    terminals = send_to_device(D, experience.terminals)
    next_states = send_to_device(D, experience.next_states)

    target_q = Qₜ(next_states)
    if haskey(t, :next_legal_actions_mask)
        target_q .+= typemin(eltype(target_q)) .* (1 .- send_to_device(D, t[:next_legal_actions_mask]))
    end

    q′ = dropdims(maximum(target_q; dims = 1), dims = 1)
    G = rewards .+ γ^update_horizon .* (1 .- terminals) .* q′

    gs = gradient(params(Q)) do
        q = Q(states)[actions]
        loss = loss_func(G, q)
        ignore() do
            learner.loss = loss
        end
        loss
    end

    update!(Q, gs)
end

function extract_experience(t::AbstractTrajectory, learner::DQNLearner)
    s = learner.stack_size
    h = learner.update_horizon
    n = learner.batch_size
    γ = learner.γ

    valid_ind_range =
        isnothing(s) ? (1:(length(t[:terminal])-h)) : (s:(length(t[:terminal])-h))
    inds = rand(learner.rng, valid_ind_range, n)
    next_inds = inds .+ h

    states = consecutive_view(t[:state], inds; n_stack = s)
    actions = consecutive_view(t[:action], inds)
    next_states = consecutive_view(t[:state], next_inds; n_stack = s)

    if haskey(t, :legal_actions_mask)
        legal_actions_mask = consecutive_view(t[:legal_actions_mask], inds)
        next_legal_actions_mask = consecutive_view(t[:next_legal_actions_mask], next_inds)
    else
        legal_actions_mask = nothing
        next_legal_actions_mask = nothing
    end

    consecutive_rewards = consecutive_view(t[:reward], inds; n_horizon = h)
    consecutive_terminals = consecutive_view(t[:terminal], inds; n_horizon = h)
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
        legal_actions_mask = legal_actions_mask,
        actions = actions,
        rewards = rewards,
        terminals = terminals,
        next_states = next_states,
        next_legal_actions_mask = next_legal_actions_mask,
    )
end
