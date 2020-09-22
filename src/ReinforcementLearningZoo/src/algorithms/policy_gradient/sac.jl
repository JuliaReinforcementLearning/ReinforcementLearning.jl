export SACPolicy, SACPolicyNetwork

using Random
using Flux
using Flux.Losses: mse
using Distributions: Normal, logpdf

# Define SAC Actor
struct SACPolicyNetwork
    pre::Chain
    mean::Chain
    log_std::Chain
end
Flux.@functor SACPolicyNetwork
(m::SACPolicyNetwork)(state) = (x = m.pre(state); (m.mean(x), m.log_std(x)))

mutable struct SACPolicy{
    BA<:NeuralNetworkApproximator,
    BC1<:NeuralNetworkApproximator,
    BC2<:NeuralNetworkApproximator,
    P,
    R<:AbstractRNG,
} <: AbstractPolicy

    policy::BA
    qnetwork1::BC1
    qnetwork2::BC2
    target_qnetwork1::BC1
    target_qnetwork2::BC2
    γ::Float32
    ρ::Float32
    α::Float32
    batch_size::Int
    start_steps::Int
    start_policy::P
    update_after::Int
    update_every::Int
    step::Int
    rng::R
end

"""
    SACPolicy(;kwargs...)

# Keyword arguments

- `policy`,
- `qnetwork1`,
- `qnetwork2`,
- `target_qnetwork1`,
- `target_qnetwork2`,
- `start_policy`,
- `γ = 0.99f0`,
- `ρ = 0.995f0`,
- `α = 0.2f0`,
- `batch_size = 32`,
- `start_steps = 10000`,
- `update_after = 1000`,
- `update_every = 50`,
- `step = 0`,
- `rng = Random.GLOBAL_RNG`,
"""
function SACPolicy(;
    policy,
    qnetwork1,
    qnetwork2,
    target_qnetwork1,
    target_qnetwork2,
    start_policy,
    γ = 0.99f0,
    ρ = 0.995f0,
    α = 0.2f0,
    batch_size = 32,
    start_steps = 10000,
    update_after = 1000,
    update_every = 50,
    step = 0,
    rng = Random.GLOBAL_RNG,
)
    copyto!(qnetwork1, target_qnetwork1)  # force sync
    copyto!(qnetwork2, target_qnetwork2)  # force sync
    SACPolicy(
        policy,
        qnetwork1,
        qnetwork2,
        target_qnetwork1,
        target_qnetwork2,
        γ,
        ρ,
        α,
        batch_size,
        start_steps,
        start_policy,
        update_after,
        update_every,
        step,
        rng,
    )
end

# TODO: handle Training/Testing mode
function (p::SACPolicy)(env)
    p.step += 1

    if p.step <= p.start_steps
        p.start_policy(env)
    else
        D = device(p.policy)
        s = get_state(env)
        s = Flux.unsqueeze(s, ndims(s) + 1)
        # trainmode:
        action = evaluate(p, s)[1][] # returns action as scalar

        # testmode:
        # if testing dont sample an action, but act deterministically by
        # taking the "mean" action
        # action = p.policy(s)[1][] # returns action as scalar
    end
end

"""
This function is compatible with a multidimensional action space.
"""
function evaluate(p::SACPolicy, state)
    μ, log_σ = p.policy(state)
    π_dist = Normal.(μ, exp.(log_σ))
    z = rand.(p.rng, π_dist)
    logp_π = sum(logpdf.(π_dist, z), dims = 1)
    logp_π -= sum((2.0f0 .* (log(2.0f0) .- z - softplus.(-2.0f0 * z))), dims = 1)
    return tanh.(z), logp_π
end

function RLBase.update!(p::SACPolicy, traj::CircularCompactSARTSATrajectory)
    length(traj[:terminal]) > p.update_after || return
    p.step % p.update_every == 0 || return

    inds = rand(p.rng, 1:(length(traj[:terminal])-1), p.batch_size)
    s = select_last_dim(traj[:state], inds)
    a = select_last_dim(traj[:action], inds)
    r = select_last_dim(traj[:reward], inds)
    t = select_last_dim(traj[:terminal], inds)
    s′ = select_last_dim(traj[:next_state], inds)

    γ, ρ, α = p.γ, p.ρ, p.α

    # !!! we have several assumptions here, need revisit when we have more complex environments
    # state is vector
    # action is scalar
    a′, log_π = evaluate(p, s′)
    q′_input = vcat(s′, a′)
    q′ = min.(p.target_qnetwork1(q′_input), p.target_qnetwork2(q′_input))

    y = r .+ γ .* (1 .- t) .* vec((q′ .- α .* log_π))

    # Train Q Networks
    a = Flux.unsqueeze(a, 1)
    q_input = vcat(s, a)

    q_grad_1 = gradient(Flux.params(p.qnetwork1)) do
        q1 = p.qnetwork1(q_input) |> vec
        mse(q1, y)
    end
    update!(p.qnetwork1, q_grad_1)
    q_grad_2 = gradient(Flux.params(p.qnetwork2)) do
        q2 = p.qnetwork1(q_input) |> vec
        mse(q2, y)
    end
    update!(p.qnetwork2, q_grad_2)

    # Train Policy
    p_grad = gradient(Flux.params(p.policy)) do
        a, log_π = evaluate(p, s)
        q_input = vcat(s, a)
        q = min.(p.qnetwork1(q_input), p.qnetwork2(q_input))
        mean(α .* log_π .- q)
    end
    update!(p.policy, p_grad)

    # polyak averaging
    for (dest, src) in zip(
        Flux.params([p.target_qnetwork1, p.target_qnetwork2]),
        Flux.params([p.qnetwork1, p.qnetwork2]),
    )
        dest .= ρ .* dest .+ (1 - ρ) .* src
    end
end
