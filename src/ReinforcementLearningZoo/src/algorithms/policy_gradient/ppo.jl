include("ppo_trajectory.jl")

using Random

export PPOLearner

"""
    PPOLearner(;kwargs)

# Keyword arguments

- `approximator`,
- `γ = 0.99f0`,
- `λ = 0.95f0`,
- `clip_range = 0.2f0`,
- `max_grad_norm = 0.5f0`,
- `n_microbatches = 4`,
- `n_epochs = 4`,
- `actor_loss_weight = 1.0f0`,
- `critic_loss_weight = 0.5f0`,
- `entropy_loss_weight = 0.01f0`,
- `rng = Random.GLOBAL_RNG`,
"""
mutable struct PPOLearner{A<:ActorCritic,R} <: AbstractLearner
    approximator::A
    γ::Float32
    λ::Float32
    clip_range::Float32
    max_grad_norm::Float32
    n_microbatches::Int
    n_epochs::Int
    actor_loss_weight::Float32
    critic_loss_weight::Float32
    entropy_loss_weight::Float32
    rng::R
    # for logging
    norm::Matrix{Float32}
    actor_loss::Matrix{Float32}
    critic_loss::Matrix{Float32}
    entropy_loss::Matrix{Float32}
    loss::Matrix{Float32}
end

function PPOLearner(;
    approximator,
    γ = 0.99f0,
    λ = 0.95f0,
    clip_range = 0.2f0,
    max_grad_norm = 0.5f0,
    n_microbatches = 4,
    n_epochs = 4,
    actor_loss_weight = 1.0f0,
    critic_loss_weight = 0.5f0,
    entropy_loss_weight = 0.01f0,
    rng = Random.GLOBAL_RNG,
)
    PPOLearner(
        approximator,
        γ,
        λ,
        clip_range,
        max_grad_norm,
        n_microbatches,
        n_epochs,
        actor_loss_weight,
        critic_loss_weight,
        entropy_loss_weight,
        rng,
        zeros(Float32, n_microbatches, n_epochs),
        zeros(Float32, n_microbatches, n_epochs),
        zeros(Float32, n_microbatches, n_epochs),
        zeros(Float32, n_microbatches, n_epochs),
        zeros(Float32, n_microbatches, n_epochs),
    )
end

function (learner::PPOLearner)(env::MultiThreadEnv)
    learner.approximator.actor(send_to_device(
        device(learner.approximator),
        get_state(env),
    )) |> send_to_host
end

function (learner::PPOLearner)(env)
    s = get_state(env)
    s = Flux.unsqueeze(s, ndims(s) + 1)
    s = send_to_device(device(learner.approximator), s)
    learner.approximator.actor(s) |> vec |> send_to_host
end

function RLBase.update!(learner::PPOLearner, t::PPOTrajectory)
    isfull(t) || return

    states = t[:state]
    actions = t[:action]
    action_log_probs = t[:action_log_prob]
    rewards = t[:reward]
    terminals = t[:terminal]
    states_plus = t[:full_state]

    rng = learner.rng
    AC = learner.approximator
    γ = learner.γ
    λ = learner.λ
    n_epochs = learner.n_epochs
    n_microbatches = learner.n_microbatches
    clip_range = learner.clip_range
    w₁ = learner.actor_loss_weight
    w₂ = learner.critic_loss_weight
    w₃ = learner.entropy_loss_weight
    D = device(AC)

    n_envs, n_rollout = size(terminals)
    @assert n_envs * n_rollout % n_microbatches == 0 "size mismatch"
    microbatch_size = n_envs * n_rollout ÷ n_microbatches

    states_flatten = flatten_batch(states)
    states_plus_flatten = flatten_batch(states_plus)
    states_plus_values =
        reshape(send_to_host(AC.critic(send_to_device(D, states_plus_flatten))), n_envs, :)
    advantages =
        generalized_advantage_estimation(rewards, states_plus_values, γ, λ; dims = 2)
    returns = advantages .+ select_last_dim(states_plus_values, 1:n_rollout)

    # TODO: normalize advantage
    for epoch in 1:n_epochs
        rand_inds = shuffle!(rng, Vector(1:n_envs*n_rollout))
        for i in 1:n_microbatches
            inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]
            s = send_to_device(D, select_last_dim(states_flatten, inds))
            if haskey(t, :legal_actions_mask)
                lam = send_to_device(
                    D,
                    select_last_dim(flatten_batch(t[:legal_actions_mask]), inds),
                )
            end
            a = vec(actions)[inds]
            r = send_to_device(D, vec(returns)[inds])
            log_p = send_to_device(D, vec(action_log_probs)[inds])
            adv = send_to_device(D, vec(advantages)[inds])

            ps = Flux.params(AC)
            gs = gradient(ps) do
                v′ = AC.critic(s) |> vec
                logit′ = AC.actor(s)
                p′ = softmax(logit′)
                log_p′ = logsoftmax(logit′)
                log_p′ₐ = log_p′[CartesianIndex.(a, 1:length(a))]

                ratio = exp.(log_p′ₐ .- log_p)
                surr1 = ratio .* adv
                surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                actor_loss = -mean(min.(surr1, surr2))
                critic_loss = mean((r .- v′) .^ 2)
                entropy_loss = -sum(p′ .* log_p′) * 1 // size(p′, 2)
                loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss

                ignore() do
                    learner.actor_loss[i, epoch] = actor_loss
                    learner.critic_loss[i, epoch] = critic_loss
                    learner.entropy_loss[i, epoch] = entropy_loss
                    learner.loss[i, epoch] = loss
                end

                loss
            end

            learner.norm[i, epoch] = clip_by_global_norm!(gs, ps, learner.max_grad_norm)
            update!(AC, gs)
        end
    end
end

function (π::QBasedPolicy{<:PPOLearner})(env::MultiThreadEnv)
    action_values = π.learner(env)
    logits = logsoftmax(action_values)
    actions = π.explorer(action_values)
    actions_log_prob = logits[CartesianIndex.(actions, 1:size(action_values, 2))]
    actions, actions_log_prob
end

(π::QBasedPolicy{<:PPOLearner})(env) = env |> π.learner |> π.explorer

function (p::RandomStartPolicy{<:QBasedPolicy{<:PPOLearner}})(env::MultiThreadEnv)
    p.num_rand_start -= 1
    if p.num_rand_start < 0
        p.policy(env)
    else
        a = p.random_policy(env)
        log_p = log.(get_prob(p.random_policy, env, a))
        a, log_p
    end
end

function (agent::Agent{<:AbstractPolicy,<:PPOTrajectory})(::Training{PreActStage}, env)
    action, action_log_prob = agent.policy(env)
    state = get_state(env)
    push!(
        agent.trajectory;
        state = state,
        action = action,
        action_log_prob = action_log_prob,
    )
    update!(agent.policy, agent.trajectory)

    # the main difference is we'd like to flush the buffer after each update!
    if isfull(agent.trajectory)
        empty!(agent.trajectory)
        push!(
            agent.trajectory;
            state = state,
            action = action,
            action_log_prob = action_log_prob,
        )
    end

    action
end

function (agent::Agent{<:AbstractPolicy,<:PPOTrajectory})(::Training{PostActStage}, env)
    push!(agent.trajectory; reward = get_reward(env), terminal = get_terminal(env))
    nothing
end

function (agent::Agent{<:AbstractPolicy,<:PPOTrajectory})(::Testing{PreActStage}, env)
    agent.policy(env)[1]  # ignore the log_prob of action
end
