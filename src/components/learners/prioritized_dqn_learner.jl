export PrioritizedDQNLearner

mutable struct PrioritizedDQNLearner{Tq<:AbstractQApproximator,Tf,Tl} <: AbstractLearner
    approximator::Tq
    target_approximator::Tq
    loss_fun::Tf
    γ::Float32
    batch_size::Int
    update_horizon::Int
    min_replay_history::Int
    update_freq::Int
    target_update_freq::Int
    update_step::Int
    loss::Tl
    default_priority::Float64
    # ??? can the code bellow simplified?
    function PrioritizedDQNLearner(
        ;
        approximator::Tq,
        target_approximator::Tq,
        loss_fun::Tf,
        γ::Float32 = 0.99f0,
        batch_size::Int = 32,
        update_horizon::Int = 1,
        min_replay_history::Int = 32,
        update_freq::Int = 1,
        target_update_freq::Int = 100,
        update_step::Int = 0,
        loss::Tl = 0.f0,
        default_priority::Float64 = 100.0,
    ) where {Tq,Tf,Tl}

        copyto!(approximator, target_approximator)  # force sync
        new{Tq,Tf,Tl}(
            approximator,
            target_approximator,
            loss_fun,
            γ,
            batch_size,
            update_horizon,
            min_replay_history,
            update_freq,
            target_update_freq,
            update_step,
            loss,
            default_priority,
        )
    end
end

function update!(learner::PrioritizedDQNLearner{<:NeuralNetworkQ}, batch)
    learner.update_step += 1
    learner.update_step % learner.update_freq == 0 || return nothing

    Q, Qₜ, γ, loss_fun, update_horizon = learner.approximator,
        learner.target_approximator,
        learner.γ,
        learner.loss_fun,
        learner.update_horizon
    states, actions, rewards, terminals, next_states = batch

    q = batch_estimate(Q, states, actions)
    q′ = dropdims(maximum(Qₜ(next_states); dims = 1), dims = 1)
    G = rewards .+ γ^update_horizon .* (1 .- terminals) .* q′

    batch_losses = loss_fun(G, q)
    priorities = (batch_losses .+ 1f-10).data
    loss = mean(batch_losses)
    learner.loss = loss.data
    update!(Q, loss)

    if learner.update_step % learner.target_update_freq == 0
        copyto!(Qₜ, Q)
    end

    priorities
end

function extract_transitions(
    buffer::CircularTurnBuffer{PRTSA},
    learner::PrioritizedDQNLearner,
)
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