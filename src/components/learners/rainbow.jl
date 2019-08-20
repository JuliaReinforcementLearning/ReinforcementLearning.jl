export RainbowLearner, update!

using Flux
using StatsBase

mutable struct RainbowLearner{Tq <: QApproximator, Tf, Tl, Ts} <: AbstractLearner{Tq}
   approximator::Tq
   target_approximator::Tq
   loss_fun::Tf
   γ::Float32
   loss::Tl
   update_freq::Int
   target_update_freq::Int
   update_step::Int
   Vₘₐₓ::Float32
   Vₘᵢₙ::Float32
   delta_z::Float32
   n_atoms::Int
   n_actions::Int
   support::Ts
end

function RainbowLearner(Q, target_Q, loss_fun;
    Vₘₐₓ,
    Vₘᵢₙ,
    n_actions,
    γ=0.99,
    loss=0.f0,
    update_freq=1,
    target_update_freq=500,
    update_step=0,
    n_atoms=51,
)
    support = range(Float32(-Vₘₐₓ), Float32(Vₘₐₓ), length=n_atoms)
    RainbowLearner(Q, target_Q, loss_fun, γ, loss, update_freq, target_update_freq, update_step, Vₘₐₓ, Vₘᵢₙ, Float32(support.step), n_atoms, n_actions, collect(support))
end

function update!(learner::RainbowLearner, consecutive_batch)
    learner.update_step += 1
    learner.update_step % learner.update_freq == 0 || return nothing

    Q, Qₜ, γ, loss_fun, n_atoms, n_actions, support, delta_z = learner.approximator, learner.target_approximator, learner.γ, learner.loss_fun, learner.n_atoms, learner.n_actions, learner.support, learner.delta_z

    states, actions, rewards, terminals, next_states = extract_SARTS(consecutive_batch, γ)

    n_step = size(consecutive_batch.states, ndims(consecutive_batch.states)-1)

    target_support = reshape(rewards, 1, :) .+ (reshape(support, :, 1) * reshape((γ ^ n_step) .* (1 .- terminals), 1, :))

    logits = reshape(Q(states), n_atoms, n_actions, :)
    select_logits = logits[:, [CartesianIndex(a, i) for (i, a) in enumerate(actions)]]
    next_logits = Qₜ(next_states).data
    next_probs = reshape(softmax(reshape(next_logits, n_atoms, :)), n_atoms, n_actions, :)
    next_q = reshape(sum(support .* next_probs, dims=1), n_actions, :)
    # next_q_argmax = argmax(cpu(next_q .+ next_legal_actions), dims=1)
    next_q_argmax = argmax(next_q, dims=1)
    next_prob_select = reshape(next_probs[:, next_q_argmax], n_atoms, :)

    target_distribution = project_distribution(target_support, next_prob_select, support, delta_z, learner.Vₘᵢₙ, learner.Vₘₐₓ)

    batch_losses = loss_fun(select_logits, target_distribution)
    updated_priorities = vec(clamp.(sqrt.(batch_losses.data .+ 1f-10), 1.f0, 1.f2))

    target_priorities = 1.0f0 ./ sqrt.(updated_priorities .+ 1f-10)
    target_priorities ./= maximum(target_priorities)
    weighted_loss = mean(target_priorities .* batch_losses)

    update!(Q, weighted_loss)
    learner.loss = weighted_loss.data

    if learner.update_step % learner.target_update_freq == 0
        copyto!(Qₜ, Q)
    end

    updated_priorities
end


function project_distribution(supports, weights, target_support, delta_z, vmin, vmax)
    batch_size, n_atoms = size(supports, 2), length(target_support)
    clampped_support = clamp.(supports, vmin, vmax)
    tiled_support = reshape(repeat(clampped_support, n_atoms), n_atoms, n_atoms, batch_size)

    projection = clamp.(1 .- abs.(tiled_support .- reshape(target_support, 1, :)) ./ delta_z, 0, 1) .* reshape(weights, n_atoms, 1, batch_size)
    reshape(sum(projection, dims=1), n_atoms, batch_size)
end