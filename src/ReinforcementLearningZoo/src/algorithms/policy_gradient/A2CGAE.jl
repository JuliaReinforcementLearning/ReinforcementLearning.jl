export A2CGAELearner

"""
    A2CGAELearner(;kwargs...)
# Keyword arguments
- `approximator`, an [`ActorCritic`](@ref) based [`NeuralNetworkApproximator`](@ref)
- `γ::Float32`, reward discount rate.
- `λ::Float32`, lambda for GAE-lambda
- `actor_loss_weight::Float32`
- `critic_loss_weight::Float32`
- `entropy_loss_weight::Float32`
"""
Base.@kwdef mutable struct A2CGAELearner{A<:ActorCritic} <: AbstractLearner
    approximator::A
    γ::Float32
    λ::Float32
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

Flux.functor(x::A2CGAELearner) = (app = x.approximator,), y -> @set x.approximator = y.app

(learner::A2CGAELearner)(env::MultiThreadEnv) =
    learner.approximator.actor(send_to_device(device(learner), state(env))) |> send_to_host

function RLBase.update!(learner::A2CGAELearner, t::CircularArraySARTTrajectory)
    length(t) == 0 && return  # in the first update, only state & action is inserted into trajectory
    learner.update_step += 1
    if learner.update_step % learner.update_freq == 0
        _update!(learner, t)
    end
end

function _update!(learner::A2CGAELearner, t::CircularArraySARTTrajectory)
    n = length(t)

    AC = learner.approximator
    to_device(x) = send_to_device(device(AC), x)
    γ = learner.γ
    λ = learner.λ
    w₁ = learner.actor_loss_weight
    w₂ = learner.critic_loss_weight
    w₃ = learner.entropy_loss_weight

    S = t[:state] |> to_device
    # (state_size..., n_thread * update_step)
    states_flattened = select_last_dim(S, 1:n) |> flatten_batch

    actions =
        t[:action] |>
        x ->
            select_last_dim(x, 1:n) |> flatten_batch |> a -> CartesianIndex.(a, 1:length(a))

    rollout_values =
        S |>
        flatten_batch |>
        AC.critic |>
        x -> reshape(x, :, n + 1) |>
        send_to_host

    advantages = generalized_advantage_estimation(
        t[:reward],
        rollout_values,
        γ,
        λ;
        dims = 2,
        terminal = t[:terminal],
    )

    gains = to_device(advantages + select_last_dim(rollout_values, 1:n))

    advantages = advantages |> flatten_batch |> to_device

    ps = Flux.params(AC)
    gs = gradient(ps) do
        logits = AC.actor(states_flattened)
        probs = softmax(logits)
        log_probs = logsoftmax(logits)
        log_probs_select = log_probs[actions]
        values = AC.critic(states_flattened)
        advantage = vec(gains) .- vec(values)
        actor_loss = -mean(log_probs_select .* advantages)
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

RLCore.check(::QBasedPolicy{<:A2CGAELearner}, ::MultiThreadEnv) = nothing
