export A2CLearner

using Flux

"""
    A2CLearner(;kwargs...)

# Keyword arguments

- `approximator`::[`ActorCritic`](@ref)
- `γ::Float32`, reward discount rate.
- `actor_loss_weight::Float32`
- `critic_loss_weight::Float32`
- `entropy_loss_weight::Float32`
"""
Base.@kwdef mutable struct A2CLearner{A<:ActorCritic} <: AbstractLearner
    approximator::A
    γ::Float32
    max_grad_norm::Union{Nothing,Float32} = nothing
    norm::Float32 = 0.f0
    actor_loss_weight::Float32
    critic_loss_weight::Float32
    entropy_loss_weight::Float32
    actor_loss::Float32 = 0.f0
    critic_loss::Float32 = 0.f0
    entropy_loss::Float32 = 0.f0
    loss::Float32 = 0.f0
end

function (learner::A2CLearner)(env::MultiThreadEnv)
    learner.approximator.actor(send_to_device(
        device(learner.approximator),
        get_state(env),
    )) |> send_to_host
end

function (learner::A2CLearner)(env)
    s = get_state(env)
    s = Flux.unsqueeze(s, ndims(s) + 1)
    s = send_to_device(device(learner.approximator), s)
    learner.approximator.actor(s) |> vec |> send_to_host
end

function RLBase.update!(learner::A2CLearner, t::AbstractTrajectory)
    isfull(t) || return

    states = t[:state]
    actions = t[:action]
    rewards = t[:reward]
    terminals = t[:terminal]
    next_state = select_last_frame(t[:next_state])

    AC = learner.approximator
    γ = learner.γ
    w₁ = learner.actor_loss_weight
    w₂ = learner.critic_loss_weight
    w₃ = learner.entropy_loss_weight
    D = device(AC)
    states = send_to_device(D, states)
    next_state = send_to_device(D, next_state)

    states_flattened = flatten_batch(states) # (state_size..., n_thread * update_step)
    actions = flatten_batch(actions)
    actions = CartesianIndex.(actions, 1:length(actions))

    next_state_values = AC.critic(next_state)
    gains = discount_rewards(
        rewards,
        γ;
        dims = 2,
        init = send_to_host(next_state_values),
        terminal = terminals,
    )
    gains = send_to_device(D, gains)

    ps = Flux.params(AC)
    gs = gradient(ps) do
        logits = AC.actor(states_flattened)
        probs = softmax(logits)
        log_probs = logsoftmax(logits)
        log_probs_select = log_probs[actions]
        values = AC.critic(states_flattened)
        advantage = vec(gains) .- vec(values)
        actor_loss = -mean(log_probs_select .* Zygote.dropgrad(advantage))
        critic_loss = mean(advantage .^ 2)
        entropy_loss = -sum(probs .* log_probs) * 1 // size(probs, 2)
        loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss
        ignore() do
            learner.actor_loss = actor_loss
            learner.critic_loss = critic_loss
            learner.entropy_loss = entropy_loss
            learner.loss = loss
        end
        loss
    end
    if !isnothing(learner.max_grad_norm)
        learner.norm = clip_by_global_norm!(gs, ps, learner.max_grad_norm)
    end
    update!(AC, gs)
end

function (agent::Agent{<:QBasedPolicy{<:A2CLearner},<:CircularCompactSARTSATrajectory})(
    ::Training{PreActStage},
    env,
)
    action = agent.policy(env)
    state = get_state(env)
    push!(agent.trajectory; state = state, action = action)
    update!(agent.policy, agent.trajectory)

    # the main difference is we'd like to flush the buffer after each update!
    if isfull(agent.trajectory)
        empty!(agent.trajectory)
        push!(agent.trajectory; state = state, action = action)
    end

    action
end
