export QBasedPolicy, get_prob

using Flux:softmax

struct QBasedPolicy{Q<:AbstractLearner, S<:AbstractActionSelector} <: AbstractPolicy
    learner::Q
    selector::S
end

(π::QBasedPolicy)(obs::Observation) = obs |> π.learner |> π.selector

"This is the default method. For some specific learners, `softmax` may be removed"
get_prob(π::QBasedPolicy, s) = s |> π.learner |> softmax

update!(π::QBasedPolicy, args...) = update!(π.learner, args...)

#####
# dispatches
#####

extract_transitions(buffer, π::QBasedPolicy) = extract_transitions(buffer, π.learner)

function extract_transitions(buffer::EpisodeTurnBuffer, ::MonteCarloLearner{T, A}) where {T, A<:AbstractQApproximator}
    if isfull(buffer)
        @views (
            states=state(buffer)[1:end-1],
            actions=action(buffer)[1:end-1],
            rewards=reward(buffer)[2:end]
        )
    else
        nothing
    end
end

function extract_transitions(buffer::EpisodeTurnBuffer, ::TDLearner{<:AbstractQApproximator, :SARSA})
    if length(buffer) > 0
        @views (
            states=state(buffer)[1:end-1],
            actions=action(buffer)[1:end-1],
            rewards=reward(buffer)[2:end],
            terminals=terminal(buffer)[2:end],
            next_states=state(buffer)[2:end],
            next_actions=action(buffer)[2:end]
        )
    else
        nothing
    end
end

function extract_transitions(buffer::EpisodeTurnBuffer, ::GradientBanditLearner)
    if length(buffer) > 0
        state(buffer)[end-1], action(buffer)[end-1], reward(buffer)[end]
    else
        nothing
    end
end

function extract_transitions(buffer::CircularTurnBuffer{RTSA}, learner::Union{QLearner, DQNLearner})
    if length(buffer) > learner.min_replay_history
        inds, consecutive_batch = sample(buffer; batch_size=learner.batch_size, n_step=learner.update_horizon)
        extract_SARTS(consecutive_batch, learner.γ)
    else
        nothing
    end
end

function extract_transitions(buffer::CircularTurnBuffer{PRTSA}, learner::Union{PrioritizedDQNLearner, RainbowLearner})
    if length(buffer) > learner.min_replay_history
        inds, consecutive_batch = sample(buffer; batch_size=learner.batch_size, n_step=learner.update_horizon)
        inds, extract_SARTS(consecutive_batch, learner.γ)
    else
        nothing
    end
end