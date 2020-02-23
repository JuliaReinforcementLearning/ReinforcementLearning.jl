export RainbowLearner

using Flux
using Zygote
using StatsBase
using Random

mutable struct RainbowLearner{
    Tq<:AbstractApproximator,
    Tt<:AbstractApproximator,
    Tf,
    Ts,
    Tss<:Union{Int,Nothing},
    R<:AbstractRNG,
} <: AbstractLearner
    approximator::Tq
    target_approximator::Tt
    loss_func::Tf
    Vₘₐₓ::Float32
    Vₘᵢₙ::Float32
    n_actions::Int
    n_atoms::Int
    support::Ts
    stack_size::Tss
    delta_z::Float32
    γ::Float32
    batch_size::Int
    update_horizon::Int
    min_replay_history::Int
    update_freq::Int
    target_update_freq::Int
    update_step::Int
    default_priority::Float64
    rng::R
end

function RainbowLearner(;
    approximator,
    target_approximator,
    loss_func,
    Vₘₐₓ,
    Vₘᵢₙ,
    n_actions,
    n_atoms = 51,
    support = collect(range(Float32(-Vₘₐₓ), Float32(Vₘₐₓ), length = n_atoms)),
    stack_size = 4,
    delta_z = Float32(support[2] - support[1]),
    γ = 0.99,
    batch_size = 32,
    update_horizon = 1,
    min_replay_history = 32,
    update_freq = 1,
    target_update_freq = 500,
    update_step = 0,
    default_priority = 100.0,
    seed = nothing,
)
    copyto!(approximator, target_approximator)  # force sync
    support = send_to_device(device(approximator), support)
    rng = MersenneTwister(seed)
    RainbowLearner(
        approximator,
        target_approximator,
        loss_func,
        Vₘₐₓ,
        Vₘᵢₙ,
        n_actions,
        n_atoms,
        support,
        stack_size,
        delta_z,
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

function (learner::RainbowLearner)(obs)
    state = send_to_device(device(learner.approximator), get_state(obs))
    logits = batch_estimate(learner.approximator, state)
    q = learner.support .* softmax(reshape(logits, :, learner.n_actions))
    # probs = vec(sum(q, dims=1)) .+ legal_action
    vec(sum(q, dims = 1)) |> send_to_host
end

function RLBase.update!(learner::RainbowLearner, batch)
    learner.update_step += 1
    learner.update_step % learner.update_freq == 0 || return nothing

    Q, Qₜ, γ, loss_func, n_atoms, n_actions, support, delta_z, update_horizon, batch_size =
        learner.approximator,
        learner.target_approximator,
        learner.γ,
        learner.loss_func,
        learner.n_atoms,
        learner.n_actions,
        learner.support,
        learner.delta_z,
        learner.update_horizon,
        learner.batch_size

    states, rewards, terminals, next_states = map(
        x -> send_to_device(device(Q), x),
        (batch.states, batch.rewards, batch.terminals, batch.next_states),
    )
    actions = CartesianIndex.(batch.actions, 1:batch_size)
    target_support =
        reshape(rewards, 1, :) .+
        (reshape(support, :, 1) * reshape((γ^update_horizon) .* (1 .- terminals), 1, :))

    updated_priorities = send_to_device(device(Q), Vector{Float32}())

    next_logits = batch_estimate(Qₜ, next_states)
    next_probs = reshape(softmax(reshape(next_logits, n_atoms, :)), n_atoms, n_actions, :)
    next_q = reshape(sum(support .* next_probs, dims = 1), n_actions, :)
    # next_q_argmax = argmax(cpu(next_q .+ next_legal_actions), dims=1)
    next_prob_select = select_best_probs(next_probs, next_q)

    target_distribution = project_distribution(
        target_support,
        next_prob_select,
        support,
        delta_z,
        learner.Vₘᵢₙ,
        learner.Vₘₐₓ,
    )

    gs = gradient(Flux.params(Q)) do
        logits = reshape(batch_estimate(Q, states), n_atoms, n_actions, :)
        select_logits = logits[:, actions]
        batch_losses = loss_func(select_logits, target_distribution)

        updated_priorities = vec(clamp.(sqrt.(batch_losses .+ 1f-10), 1.f0, 1.f2))
        target_priorities = 1.0f0 ./ sqrt.(updated_priorities .+ 1f-10)
        normalized_target_priorities = target_priorities ./ maximum(target_priorities)

        mean(Zygote.dropgrad(normalized_target_priorities) .* batch_losses)
    end

    update!(Q, gs)

    if learner.update_step % learner.target_update_freq == 0
        copyto!(Qₜ, Q)
    end

    updated_priorities |> send_to_host
end

@inline function select_best_probs(probs, q)
    q_argmax = argmax(q, dims = 1)
    prob_select = @inbounds probs[:, q_argmax] # !!! without @inbounds it would be really slow
    reshape(prob_select, :, length(q_argmax))
end

function project_distribution(supports, weights, target_support, delta_z, vmin, vmax)
    batch_size, n_atoms = size(supports, 2), length(target_support)
    clampped_support = clamp.(supports, vmin, vmax)
    tiled_support = reshape(
        repeat(clampped_support; outer = (n_atoms, 1)),
        n_atoms,
        n_atoms,
        batch_size,
    )

    projection =
        clamp.(
            1 .- abs.(tiled_support .- reshape(target_support, 1, :)) ./ delta_z,
            0,
            1,
        ) .* reshape(weights, n_atoms, 1, batch_size)
    reshape(sum(projection, dims = 1), n_atoms, batch_size)
end

function RLBase.extract_experience(t::AbstractTrajectory, learner::RainbowLearner)
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

        # 3. make sure that we only consider experiences in current episode for rewards and terminals
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

function RLBase.update!(p::QBasedPolicy{<:RainbowLearner}, t::AbstractTrajectory)
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
    agent::Agent{<:QBasedPolicy{<:RainbowLearner},<:CircularCompactPSARTSATrajectory}
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
