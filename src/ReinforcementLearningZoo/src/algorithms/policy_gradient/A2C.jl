export A2CLearner, ActorCritic

using Flux

"""
    ActorCritic(actor, critic)

The `actor` part must return a **normalized** vector representing the action values,
and the `critic` part must return a state value.
"""
Base.@kwdef struct ActorCritic{A,C}
    actor::A
    critic::C
end

Flux.@functor ActorCritic

(m::ActorCritic)(s::AbstractArray, ::Val{:Q}) = m.actor(s)
(m::ActorCritic)(s::AbstractArray, ::Val{:V}) = m.critic(s)

"""
    A2CLearner(;kwargs...)

# Keyword arguments

- `approximator`, an [`ActorCritic`](@ref) based [`NeuralNetworkApproximator`](@ref)
- `γ::Float32`, reward discount rate.
- `actor_loss_weight::Float32`
- `critic_loss_weight::Float32`
- `entropy_loss_weight::Float32`
- `update_freq=1`, it **must** be the same with the length of the `CircularCompactSARTSATrajectory`.
- `update_step=0`, an internal counter to record how many times the `update!(learner, experience)` method has been called.
"""
Base.@kwdef mutable struct A2CLearner{A} <: AbstractLearner
    approximator::A
    γ::Float32
    actor_loss_weight::Float32
    critic_loss_weight::Float32
    entropy_loss_weight::Float32
    update_freq::Int = 1
    update_step::Int = 0
end

(learner::A2CLearner)(obs::BatchObs) =
    learner.approximator(
        send_to_device(device(learner.approximator), get_state(obs)),
        Val(:Q),
    ) |> send_to_host

function RLBase.update!(learner::A2CLearner, experience)
    learner.update_step += 1
    learner.update_step % learner.update_freq == 0 || return nothing

    AC = learner.approximator
    γ = learner.γ
    w₁ = learner.actor_loss_weight
    w₂ = learner.critic_loss_weight
    w₃ = learner.entropy_loss_weight
    states, actions, rewards, terminals, next_state = experience
    states = send_to_device(device(AC), states)
    next_state = send_to_device(device(AC), next_state)

    states_flattened = flatten_batch(states) # (state_size..., n_thread * update_step)
    actions = flatten_batch(actions)
    actions = CartesianIndex.(actions, 1:length(actions))

    next_state_values = AC(next_state, Val(:V))
    gains = discount_rewards(
        rewards,
        γ;
        dims = 2,
        init = send_to_host(next_state_values),
        terminal = terminals,
    )
    gains = send_to_device(device(AC), gains)

    gs = gradient(Flux.params(AC)) do
        probs = AC(states_flattened, Val(:Q))
        log_probs = log.(probs)
        log_probs_select = probs[actions]
        values = AC(states_flattened, Val(:V))
        advantage = vec(gains) .- vec(values)
        actor_loss = -mean(log_probs_select .* Zygote.dropgrad(advantage))
        critic_loss = mean(advantage .^ 2)
        entropy_loss = sum(probs .* log_probs) * 1 // size(probs, 2)
        loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss
        loss
    end
    update!(AC, gs)
end

function RLBase.extract_experience(t::CircularCompactSARTSATrajectory, learner::A2CLearner)
    if isfull(t)
        (
            states = get_trace(t, :state),
            actions = get_trace(t, :action),
            rewards = get_trace(t, :reward),
            terminals = get_trace(t, :terminal),
            next_state = select_last_frame(get_trace(t, :next_state)),
        )
    else
        nothing
    end
end
