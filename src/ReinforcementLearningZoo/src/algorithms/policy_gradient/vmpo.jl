export VMPOPolicy, VMPOTrajectory

using LinearAlgebra: ⋅  # dot product
using Zygote: ignore, dropgrad

## trajectory

const VMPOTrajectory = CircularArraySARTTrajectory

## policy definition

"""
    VMPOPolicy(;kwargs)

V-MPO, an on-policy adaptation of Maximum a Posteriori Policy Optimization (MPO)
that performs policy iteration based on a learned state-value function.

# Keyword arguments

- `approximator`: an [`ActorCritic`](@ref) based on [`NeuralNetworkApproximator`](@ref)
- `update_freq`: update policy every n timesteps
- `γ = 0.99f0`: discount factor
- `ϵ_η = 0.02f0`: temperature η hyperparameter
- `ϵ_α = 0.1f0`: Lagrange multiplier α (discrete) hyperparameter
- `ϵ_αμ = 0.005f0`: Lagrange multiplier α_mu (continuous) hyperparameter
- `ϵ_ασ = 0.00005f0`: Lagrange multiplier α_σ (continuous) hyperparameter
- `n_epochs = 8`: update policy for n epochs
- `dist = Categorical`: `Categorical` - discrete, `Normal` - continuous
- `rng = Random.GLOBAL_RNG`

By default, `dist` is set to `Categorical`, which means it will only work
on environments of discrete actions. To work with environments of continuous
actions `dist` should be set to `Normal` and the `actor` in the `approximator`
should be a `GaussianNetwork`. This algorithm only supports one-dimensional
action space for now.

# Ref paper

[V-MPO: On-Policy Maximum a Posteriori Policy Optimization for Discrete and Continuous Control](https://arxiv.org/abs/1909.12238)
"""
mutable struct VMPOPolicy{A<:ActorCritic,D,R} <: AbstractPolicy
    approximator::A
    γ::Float32
    ϵ_η::Float32
    ϵ_α::Float32
    ϵ_αμ::Float32
    ϵ_ασ::Float32
    η::Vector{Float32}  # 1-element array for temperature η
    α::Vector{Float32}  # 3-element array for Lagrange multipliers: [α, αμ, ασ]
    n_epochs::Int
    update_freq::Int
    update_step::Int
    rng::R
end

function VMPOPolicy(;
    approximator,
    update_freq,
    γ=0.99f0,
    ϵ_η=0.02f0,
    ϵ_α=0.1f0,
    ϵ_αμ=0.005f0,
    ϵ_ασ=0.00005f0,
    n_epochs=8,
    dist=Categorical,
    rng=Random.GLOBAL_RNG
)
    VMPOPolicy{typeof(approximator),dist,typeof(rng)}(
        approximator,
        γ,
        ϵ_η,
        ϵ_α,
        ϵ_αμ,
        ϵ_ασ,
        [1.0f0],                # [η]
        [1.0f0, 1.0f0, 1.0f0],  # [α, αμ, ασ]
        n_epochs,
        update_freq,
        0,
        rng,
    )
end

## policy environment interactions

# discrete action

function RLBase.prob(policy::VMPOPolicy{<:ActorCritic,Categorical}, env::AbstractEnv)
    s = send_to_device(device(policy.approximator), state(env))
    p = policy.approximator.actor(s) |> softmax |> send_to_host
    Categorical(p; check_args=false)
end

function (agent::Agent{<:VMPOPolicy{<:ActorCritic,Categorical}})(env::AbstractEnv)
    dist = prob(agent.policy, env)
    rand(agent.policy.rng, dist)
end

function (policy::VMPOPolicy{<:ActorCritic,Categorical})(
    state::AbstractArray,
    action::AbstractArray,
)
    p = policy.approximator.actor(state) |> softmax
    idx = ignore() do
        CartesianIndex.(action, 1:length(action))
    end
    action_log_prob = log.(p)[idx]
    p, action_log_prob
end

# continuous action

function RLBase.prob(
    policy::VMPOPolicy{<:ActorCritic{<:GaussianNetwork},Normal},
    env::AbstractEnv,
)
    s = send_to_device(device(policy.approximator), state(env))
    μ, logσ = policy.approximator.actor(agent.policy.rng, s)
    Normal(μ, exp(logσ))
end

function (agent::Agent{<:VMPOPolicy{<:ActorCritic{<:GaussianNetwork},Normal}})(
    env::AbstractEnv,
)
    s = send_to_device(device(agent.policy.approximator), state(env))
    # the action is an output of GaussianNetwork which is normalised by tanh(),
    # we increase its stability by limiting it to [-1 + eps, 1 - eps]
    a = agent.policy.approximator.actor(agent.policy.rng, s, is_sampling=true)
    m = one(eltype(a)) - eps(eltype(a))
    clamp.(a, -m, m) |> send_to_host |> first
end

function (policy::VMPOPolicy{<:ActorCritic{<:GaussianNetwork},Normal})(
    state::AbstractArray,
    action::AbstractArray,
)
    μ, logσ = policy.approximator.actor(policy.rng, state)
    action = atanh.(action)
    μ, logσ, normlogpdf(μ, exp.(logσ), reshape(action, size(μ)...))
end

## update policy

function RLBase.update!(p::VMPOPolicy, t::VMPOTrajectory, env::AbstractEnv, ::PreActStage)
    # in the first update, only state & action are inserted into trajectory
    length(t) == 0 && return

    p.update_step += 1
    if p.update_step % p.update_freq == 0
        _update!(p, t)
    end
end

function _update!(p::VMPOPolicy, t::VMPOTrajectory)
    AC = p.approximator
    D = device(AC)
    s = send_to_device(D, t[:state][:, 1:end-1])  # drop the last extra state
    a = send_to_device(D, t[:action][1:end-1])  # drop the last extra action
    is_discrete = isa(p, VMPOPolicy{<:ActorCritic,Categorical})
    π_old = is_discrete ? AC.actor(s) |> softmax : nothing
    μ_old, σ_old = is_discrete ? (nothing, nothing) : begin
        μ, logσ = AC.actor(s)
        μ, exp.(logσ)
    end

    # normalise rewards based on each trajectory
    inds = vcat(0, findall(t[:terminal]), length(t[:terminal]))
    rewards =
        map(
            Flux.normalise,
            view.(Ref(t[:reward]), (:).(inds[1:end-1] .+ 1, inds[2:end])),
        ) |>
        Iterators.flatten |>
        collect

    # calculate discounted rewards
    rewards = discount_rewards(rewards, p.γ, terminal=t[:terminal])
    rewards = send_to_device(D, rewards)

    # calculate advantages and sample top half of it
    advantages = rewards .- AC.critic(s)[1, :]
    top_k_idx = sortperm(advantages |> send_to_host, rev=true)[1:length(advantages)÷2]
    top_k_advs = advantages[top_k_idx]

    for epoch in 1:p.n_epochs
        ps = Flux.params(AC, p.η, p.α)
        gs = gradient(ps) do
            η, α, αμ, ασ = p.η[1], p.α...
            π, logπ = is_discrete ? p(s, a) : (nothing, nothing)
            μ, σ, logπ =
                is_discrete ? (nothing, nothing, logπ) : begin
                    μ, logσ, logπ = p(s, a)
                    μ, exp.(logσ), logπ
                end

            # critic loss
            v = AC.critic(s)[1, :]
            Lᵥ = Flux.Losses.mse(v, rewards) / 2

            # actor loss
            ψ = softmax(top_k_advs / dropgrad(η))
            Lπ = -ψ ⋅ logπ[top_k_idx]

            # loss for temperature η
            Lη = η * p.ϵ_η + η * log(mean(exp.(top_k_advs / η)))

            # loss for Lagrange multiplier α
            Lα = if is_discrete
                KL = Flux.kldivergence(π_old, π, agg=identity)
                mean(α * (p.ϵ_α .- dropgrad(KL)) + dropgrad(α) * KL)
            else
                KLμ = 0.5f0 * (μ .- μ_old) .^ 2 ./ (σ .^ 2)
                Lαμ = mean(αμ * (p.ϵ_αμ .- dropgrad(KLμ)) + dropgrad(αμ) * KLμ)
                KLσ = 0.5f0 * ((σ_old ./ σ) .^ 2 .- 1 .+ 2 * log.(σ ./ σ_old))
                Lασ = mean(ασ * (p.ϵ_ασ .- dropgrad(KLσ)) + dropgrad(ασ) * KLσ)
                Lαμ + Lασ
            end

            loss = Lᵥ + Lπ + Lη + Lα
        end

        # skip if gradient contains NaN or Inf
        if any(x -> !isnothing(x) && any(y -> isnan(y) || isinf(y), x), gs)
            break
        end

        Flux.Optimise.update!(AC.optimizer, ps, gs)

        p.η = clamp.(p.η, 1e-8, p.η)
        p.α = clamp.(p.α, 1e-8, p.α)
    end
end
