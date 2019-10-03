export RainbowLearner, update!

using Flux, Zygote
using StatsBase

Base.@kwdef mutable struct RainbowLearner{Tq<:AbstractQApproximator,Tf,Ts} <: AbstractLearner
    approximator::Tq
    target_approximator::Tq
    loss_fun::Tf
    Vₘₐₓ::Float32
    Vₘᵢₙ::Float32
    n_actions::Int
    n_atoms::Int = 51
    support::Ts = collect(range(Float32(-Vₘₐₓ), Float32(Vₘₐₓ), length = n_atoms))
    delta_z::Float32 = Float32(support[2] - support[1])
    γ::Float32 = 0.99
    loss::Float32 = 0.f0
    batch_size::Int = 32
    update_horizon::Int = 1
    min_replay_history::Int = 32
    update_freq::Int = 1
    target_update_freq::Int = 500
    update_step::Int = 0
    default_priority::Float64 = 100.0

    function RainbowLearner(approximator::Tq, target_approximator::Tq, loss_fun::Tf, Vₘₐₓ::Float32, Vₘᵢₙ::Float32, n_actions::Int, n_atoms::Int, support::Ts, args...) where {Tq,Tf, Ts}
        copyto!(approximator, target_approximator)  # force sync
        new{Tq,Tf, Ts}(approximator, target_approximator, loss_fun, Vₘₐₓ, Vₘᵢₙ, n_actions, n_atoms, support, args...)
    end
end

function (learner::RainbowLearner)(obs::Observation)
    logits = obs |> get_state |> learner.approximator
    q = learner.support .* softmax(reshape(logits, :, learner.n_actions))
    # probs = vec(sum(q, dims=1)) .+ legal_action
    vec(sum(q, dims = 1))
end

function update!(learner::RainbowLearner, batch)
    learner.update_step += 1
    learner.update_step % learner.update_freq == 0 || return nothing

    Q, Qₜ, γ, loss_fun, n_atoms, n_actions, support, delta_z, update_horizon, batch_size = learner.approximator,
        learner.target_approximator,
        learner.γ,
        learner.loss_fun,
        learner.n_atoms,
        learner.n_actions,
        learner.support,
        learner.delta_z,
        learner.update_horizon,
        learner.batch_size

    states, rewards, terminals, next_states = map(x->to_device(Q, x), (batch.states, batch.rewards, batch.terminals, batch.next_states))
    actions = CartesianIndex.(batch.actions, 1:batch_size)

    target_support = reshape(rewards, 1, :) .+
                     (reshape(support, :, 1) *
                      reshape((γ^update_horizon) .* (1 .- terminals), 1, :))

    updated_priorities = Vector{Float32}()

    loss, back = Flux.pullback(Q.params) do 
        logits = reshape(batch_estimate(Q, states), n_atoms, n_actions, :)
        select_logits = logits[:, actions]
        next_logits = batch_estimate(Qₜ, next_states)
        next_probs = reshape(softmax(reshape(next_logits, n_atoms, :)), n_atoms, n_actions, :)
        next_q = reshape(sum(support .* next_probs, dims = 1), n_actions, :)
        # next_q_argmax = argmax(cpu(next_q .+ next_legal_actions), dims=1)
        next_q_argmax = argmax(next_q, dims = 1)
        next_prob_select = reshape(next_probs[:, next_q_argmax], n_atoms, :)

        target_distribution = project_distribution(
            target_support,
            next_prob_select,
            support,
            delta_z,
            learner.Vₘᵢₙ,
            learner.Vₘₐₓ,
        )

        batch_losses = loss_fun(select_logits, target_distribution)
        updated_priorities = vec(clamp.(sqrt.(Zygote.dropgrad(batch_losses) .+ 1f-10), 1.f0, 1.f2))

        target_priorities = 1.0f0 ./ sqrt.(updated_priorities .+ 1f-10)
        target_priorities ./= maximum(target_priorities)

        mean(Zygote.dropgrad(target_priorities) .* batch_losses)
    end

    learner.loss = loss
    update!(Q, back(loss))

    if learner.update_step % learner.target_update_freq == 0
        copyto!(Qₜ, Q)
    end

    updated_priorities
end

function project_distribution(supports, weights, target_support, delta_z, vmin, vmax)
    batch_size, n_atoms = size(supports, 2), length(target_support)
    clampped_support = clamp.(supports, vmin, vmax)
    tiled_support = reshape(repeat(clampped_support, n_atoms), n_atoms, n_atoms, batch_size)

    projection = clamp.(
        1 .- abs.(tiled_support .- reshape(target_support, 1, :)) ./ delta_z,
        0,
        1,
    ) .* reshape(weights, n_atoms, 1, batch_size)
    reshape(sum(projection, dims = 1), n_atoms, batch_size)
end

function extract_transitions(buffer::CircularTurnBuffer{PRTSA}, learner::RainbowLearner)
    if length(buffer) > learner.min_replay_history
        inds, consecutive_batch = sample(
            buffer;
            batch_size = learner.batch_size,
            n_step = learner.update_horizon,
        )
        inds, extract_SARTS(consecutive_batch, learner.γ)
    else
        nothing
    end
end