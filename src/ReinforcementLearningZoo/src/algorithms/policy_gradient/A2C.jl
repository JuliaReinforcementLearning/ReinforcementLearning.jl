export A2CLearner

"""
    A2CLearner(;kwargs...)

# Keyword arguments

- `approximator`::[`ActorCritic`](@ref)
- `γ::Float32`, reward discount rate.
- `actor_loss_weight::Float32`
- `critic_loss_weight::Float32`
- `entropy_loss_weight::Float32`
- `update_freq::Int`, usually set to the same with the length of trajectory.
"""
Base.@kwdef mutable struct A2CLearner{A<:ActorCritic} <: AbstractLearner
    approximator::A
    γ::Float32
    max_grad_norm::Union{Nothing,Float32} = nothing
    actor_loss_weight::Float32
    critic_loss_weight::Float32
    entropy_loss_weight::Float32
    update_freq::Int
    update_step::Int = 0
    # for logging
    actor_loss::Float32 = 0.0f0
    critic_loss::Float32 = 0.0f0
    entropy_loss::Float32 = 0.0f0
    loss::Float32 = 0.0f0
    norm::Float32 = 0.0f0
end

Flux.functor(x::A2CLearner) = (app = x.approximator,), y -> @set x.approximator = y.app

function (learner::A2CLearner)(env::MultiThreadEnv)
    learner.approximator.actor(send_to_device(device(learner), state(env))) |> send_to_host
end

function (learner::A2CLearner)(env)
    s = state(env)
    s = Flux.unsqueeze(s, ndims(s) + 1)
    s = send_to_device(device(learner), s)
    learner.approximator.actor(s) |> vec |> send_to_host
end

function RLBase.update!(learner::A2CLearner, t::CircularArraySARTTrajectory)
    length(t) == 0 && return  # in the first update, only state & action is inserted into trajectory
    learner.update_step += 1
    if learner.update_step % learner.update_freq == 0
        _update!(learner, t)
    end
end

function _update!(learner::A2CLearner, t::CircularArraySARTTrajectory)
    n = length(t)

    AC = learner.approximator
    γ = learner.γ
    w₁ = learner.actor_loss_weight
    w₂ = learner.critic_loss_weight
    w₃ = learner.entropy_loss_weight
    to_device = x -> send_to_device(device(AC), x)

    S = t[:state] |> to_device
    states = select_last_dim(S, 1:n)
    states_flattened = flatten_batch(states) # (state_size..., n_thread * update_freq)

    actions = select_last_dim(t[:action], 1:n)
    actions = flatten_batch(actions)
    actions = CartesianIndex.(actions, 1:length(actions))

    next_state_values = S |> select_last_frame |> AC.critic |> send_to_host

    gains =
        discount_rewards(
            t[:reward],
            γ;
            dims = 2,
            init = send_to_host(next_state_values),
            terminal = t[:terminal],
        ) |> to_device

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

RLCore.check(::QBasedPolicy{<:A2CLearner}, ::MultiThreadEnv) = nothing
