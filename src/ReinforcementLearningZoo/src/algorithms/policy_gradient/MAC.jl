export MACLearner

"""
    MACLearner(;kwargs...)

[Mean Actor-Critic](https://arxiv.org/pdf/1709.00503.pdf)

# Keyword arguments

- `approximator`::[`ActorCritic`](@ref)
- `γ::Float32`, reward discount rate
-  `bootstrap::bool`, if false then Q function is approximated using monte carlo returns.
"""

Base.@kwdef mutable struct MACLearner{A<:ActorCritic} <: AbstractLearner
    approximator::A
    γ::Float32
    max_grad_norm::Union{Nothing,Float32} = nothing
    norm::Float32 = 0.0f0
    actor_loss::Float32 = 0.0f0
    critic_loss::Float32 = 0.0f0
    loss::Float32 = 0.0f0
    bootstrap::Bool = true
    update_freq::Int
    update_step::Int = 0
end

Flux.functor(x::MACLearner) = (app = x.approximator,), y -> @set x.approximator = y.app

function (learner::MACLearner)(env::MultiThreadEnv)
    learner.approximator.actor(send_to_device(device(learner.approximator), state(env))) |>
    send_to_host
end

function (learner::MACLearner)(env)
    s = state(env)
    s = Flux.unsqueeze(s, ndims(s) + 1)
    s = send_to_device(device(learner.approximator), s)
    learner.approximator.actor(s) |> vec |> send_to_host
end

function RLBase.update!(
    learner::MACLearner,
    t::CircularArraySARTTrajectory,
    ::AbstractEnv,
    ::PreActStage,
)
    length(t) == 0 && return  # in the first update, only state & action is inserted into trajectory
    learner.update_step += 1
    if learner.update_step % learner.update_freq == 0
        _update!(learner, t)
    end
end


function _update!(learner::MACLearner, t::CircularArraySARTTrajectory)
    n = length(t)
    states = select_last_dim(t[:state], 1:n)
    actions = select_last_dim(t[:action], 1:n)
    rewards = t[:reward]
    terminals = t[:terminal]

    AC = learner.approximator
    γ = learner.γ
    D = device(AC)

    states = send_to_device(D, states)
    states_flattened = flatten_batch(states) # (state_size..., n_thread * update_freq)


    actions = flatten_batch(actions)
    actions = CartesianIndex.(actions, 1:length(actions))

    if learner.bootstrap
        next_state = select_last_frame(t[:state])
        next_state = send_to_device(D, next_state)
        next_state_values = AC.critic(next_state)

        gains = discount_rewards(
            rewards,
            γ;
            dims = 2,
            init = send_to_host(next_state_values),
            terminal = terminals,
        )
        gains = send_to_device(D, gains)
    else
        next_state_flattened = flatten_batch(select_last_dim(t[:state], 2:n+1))
        next_state_flattened = send_to_device(D, next_state_flattened)
        rewards_flattened = flatten_batch(rewards)
        rewards_flattened = send_to_device(D, rewards_flattened)
    end

    action_values = AC.critic(states_flattened)

    ps1 = Flux.params(AC.actor)
    gs1 = gradient(ps1) do
        logits = AC.actor(states_flattened)
        probs = softmax(logits)
        actor_loss = -mean(sum((probs .* Zygote.dropgrad(action_values)), dims = 1))
        loss = actor_loss
        ignore() do
            learner.actor_loss = actor_loss
        end
        loss
    end
    if !isnothing(learner.max_grad_norm)
        learner.norm = clip_by_global_norm!(gs1, ps1, learner.max_grad_norm)
    end
    update!(AC.actor, gs1)

    ps2 = Flux.params(AC.critic)
    gs2 = gradient(ps2) do
        if learner.bootstrap
            critic_loss = mean((vec(gains) .- vec(action_values[actions])) .^ 2)
        else
            next_state_values = AC.critic(next_state_flattened)
            target_action_values =
                vec(rewards_flattened) .+
                γ * vec(
                    Zygote.dropgrad(
                        sum(
                            next_state_values .* softmax(AC.actor(next_state_flattened)),
                            dims = 1,
                        ),
                    ),
                )
            critic_loss =
                mean((vec(target_action_values) .- vec(action_values[actions])) .^ 2)
        end

        loss = critic_loss
        ignore() do
            learner.critic_loss = critic_loss
        end
        loss
    end
    if !isnothing(learner.max_grad_norm)
        learner.norm = clip_by_global_norm!(gs2, ps2, learner.max_grad_norm)
    end
    update!(AC.critic, gs2)
end

RLCore.check(::QBasedPolicy{<:MACLearner}, ::MultiThreadEnv) = nothing
