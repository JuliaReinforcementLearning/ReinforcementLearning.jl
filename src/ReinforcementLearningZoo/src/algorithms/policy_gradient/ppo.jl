include("ppo_trajectory.jl")

using Random
using Distributions: Categorical, Normal, logpdf
using StructArrays

export PPOPolicy

"""
    PPOPolicy(;kwargs)

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
- `dist = Categorical`,
- `rng = Random.GLOBAL_RNG`,

By default, `dist` is set to `Categorical`, which means it will only works
on environments of discrete actions. To work with environments of
"""
mutable struct PPOPolicy{A<:ActorCritic,D,R} <: AbstractPolicy
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

function PPOPolicy(;
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
    dist = Categorical,
    rng = Random.GLOBAL_RNG,
)
    PPOPolicy{typeof(approximator),dist,typeof(rng)}(
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

function RLBase.get_prob(p::PPOPolicy{<:ActorCritic{<:NeuralNetworkApproximator{<:GaussianNetwork}}, Normal}, state::AbstractArray)
    p.approximator.actor(send_to_device(
        device(p.approximator),
        state,
    )) |> send_to_host |> StructArray{Normal}
end

function RLBase.get_prob(p::PPOPolicy{<:ActorCritic, Categorical}, state::AbstractArray)
    logits = p.approximator.actor(send_to_device(
        device(p.approximator),
        state,
    )) |> softmax |> send_to_host
    [Categorical(x;check_args=false) for x in eachcol(logits)]
end

RLBase.get_prob(p::PPOPolicy, env::MultiThreadEnv) = get_prob(p, get_state(env))

function RLBase.get_prob(p::PPOPolicy, env::AbstractEnv)
    s = get_state(env)
    s = Flux.unsqueeze(s, ndims(s) + 1)
    get_prob(p, s)[1]
end

(p::PPOPolicy)(env::MultiThreadEnv) = rand.(p.rng, get_prob(p, env))
(p::PPOPolicy)(env::AbstractEnv) = rand(p.rng, get_prob(p, env))

function RLBase.update!(p::PPOPolicy, t::PPOTrajectory)
    isfull(t) || return

    states = t[:state]
    actions = t[:action]
    action_log_probs = t[:action_log_prob]
    rewards = t[:reward]
    terminals = t[:terminal]
    states_plus = t[:full_state]

    rng = p.rng
    AC = p.approximator
    γ = p.γ
    λ = p.λ
    n_epochs = p.n_epochs
    n_microbatches = p.n_microbatches
    clip_range = p.clip_range
    w₁ = p.actor_loss_weight
    w₂ = p.critic_loss_weight
    w₃ = p.entropy_loss_weight
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
                if AC.actor isa NeuralNetworkApproximator{<:GaussianNetwork}
                    μ, σ = AC.actor(s)
                    log_p′ₐ = normlogpdf(μ, σ, a)
                    entropy_loss = mean((log(2.0f0π)+1)/2 .+ log.(σ))
                else
                    # actor is assumed to return discrete logits
                    logit′ = AC.actor(s)
                    p′ = softmax(logit′)
                    log_p′ = logsoftmax(logit′)
                    log_p′ₐ = log_p′[CartesianIndex.(a, 1:length(a))]
                    entropy_loss = -sum(p′ .* log_p′) * 1//size(p′, 2)
                end

                ratio = exp.(log_p′ₐ .- log_p)
                surr1 = ratio .* adv
                surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                actor_loss = -mean(min.(surr1, surr2))
                critic_loss = mean((r .- v′) .^ 2)
                loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss

                ignore() do
                    p.actor_loss[i, epoch] = actor_loss
                    p.critic_loss[i, epoch] = critic_loss
                    p.entropy_loss[i, epoch] = entropy_loss
                    p.loss[i, epoch] = loss
                end

                loss
            end

            p.norm[i, epoch] = clip_by_global_norm!(gs, ps, p.max_grad_norm)
            update!(AC, gs)
        end
    end
end

function (agent::Agent{<:Union{PPOPolicy, RandomStartPolicy{<:PPOPolicy}}})(::Training{PreActStage}, env::MultiThreadEnv)
    state = get_state(env)
    dist = get_prob(agent.policy, env)

    # currently RandomPolicy returns a Matrix instead of a (vector of) distribution.
    if dist isa Matrix{<:Number}
        dist = [Categorical(x;check_args=false) for x in eachcol(dist)]
    elseif dist isa Vector{<:Vector{<:Number}}
        dist = [Categorical(x;check_args=false) for x in dist]
    end

    # !!! a little ugly
    rng = if agent.policy isa PPOPolicy
        agent.policy.rng
    elseif agent.policy isa RandomStartPolicy
        agent.policy.policy.rng
    end

    action = [rand(rng, d) for d in dist]
    action_log_prob = [logpdf(d, a) for (d, a) in zip(dist, action)]
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
