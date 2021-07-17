export CQLPolicy

mutable struct CQLPolicy{
    BA<:NeuralNetworkApproximator,
    BC1<:NeuralNetworkApproximator,
    BC2<:NeuralNetworkApproximator,
    R<:AbstractRNG,
} <: AbstractPolicy
    policy::BA
    qnetwork1::BC1
    qnetwork2::BC2
    target_qnetwork1::BC1
    target_qnetwork2::BC2
    γ::Float32
    τ::Float32
    α::Float32
    η::Float32
    ηₐ::float32
    batch_size::Int
    automatic_entropy_tuning::Bool
    lr_alpha::Float32
    target_entropy::Float32
    step::Int
    rng::R
    # Logging
    reward_term::Float32
    entropy_term::Float32
end

"""
    CQLPolicy(;kwargs...)

# Keyword arguments

- `policy`,
- `qnetwork1`,
- `qnetwork2`,
- `target_qnetwork1`,
- `target_qnetwork2`,
- `γ = 0.99f0`,
- `τ = 0.005f0`,
- `α = 0.2f0`,
- `η = 3f-4`, Hyperparameter for critics, based on the recommendations of the paper
- `ηₐ = 3f-4`, Hyperparameter for the actor, based on the recommendations of the paper
- `batch_size = 256`,
- `automatic_entropy_tuning::Bool = false`, whether to automatically tune the entropy.
- `lr_alpha::Float32 = 0.003f0`, learning rate of tuning entropy.
- `action_dims = 0`, the dimension of the action. if `automatic_entropy_tuning = true`, must enter this parameter.
- `step = 0`,
- `rng = Random.GLOBAL_RNG`,

`policy` is expected to output a tuple `(μ, logσ)` of mean and
log standard deviations for the desired action distributions, this
can be implemented using a `GaussianNetwork` in a `NeuralNetworkApproximator`.

Implemented based on https://arxiv.org/abs/2006.04779
"""

function CQLPolicy(;
    policy,
    qnetwork1,
    qnetwork2,
    target_qnetwork1,
    target_qnetwork2,
    γ = 0.99f0,
    τ = 0.005f0,
    α = 0.2f0,
    η = 3f-4,
    ηₐ= 3f-5,
    batch_size = 32,
    automatic_entropy_tuning = true,
    lr_alpha = 0.003f0,
    action_dims = 0,
    step = 0,
    rng = Random.GLOBAL_RNG,
)
    copyto!(qnetwork1, target_qnetwork1)  # force sync
    copyto!(qnetwork2, target_qnetwork2)  # force sync
    if automatic_entropy_tuning
        @assert action_dims != 0
    end
    CQLPolicy(
        policy,
        qnetwork1,
        qnetwork2,
        target_qnetwork1,
        target_qnetwork2,
        γ,
        τ,
        α,
        η,
        ηₐ,
        batch_size,
        automatic_entropy_tuning,
        lr_alpha,
        Float32(-action_dims),
        step,
        rng,
        0f0,
        0f0,
    )
end

# CQL(H) loss  until CQL loss is impleented in common.jl for offline algorithms

function CQL_loss(q_value::Vector{T}, action_val::Vector{R}) where {T, R}
    mean(log.(sum(exp.(q_value), dims=1)) .- action_val)
end

# TODO: handle Training/Testing mode
function (p::CQLPolicy)(env)
    p.step += 1

    D = device(p.policy)
    s = state(env)
    s = Flux.unsqueeze(s, ndims(s) + 1)
    # trainmode:
    action = dropdims(evaluate(p, s)[1], dims=2) # Single action vec, drop second dim

    # testmode:
    # if testing dont sample an action, but act deterministically by
    # taking the "mean" action
    # action = dropdims(p.policy(s)[1], dims=2) 
end

"""
This function is compatible with a multidimensional action space.
"""
function evaluate(p::CQLPolicy, state)
    μ, logσ = p.policy(state)
    π_dist = Normal.(μ, exp.(logσ))
    z = rand.(p.rng, π_dist)
    logp_π = sum(logpdf.(π_dist, z), dims = 1)
    logp_π -= sum((2.0f0 .* (log(2.0f0) .- z - softplus.(-2.0f0 * z))), dims = 1)
    return tanh.(z), logp_π
end


function RLBase.update!(p::CQLPolicy, batch::NamedTuple{SARTS})
    s, a, r, t, s′ = send_to_device(device(p.qnetwork1), batch)

    γ, τ, α, η, ηₐ = p.γ, p.τ, p.α, p.η, p.ηₐ

    a′, log_π = evaluate(p, s′)
    q′_input = vcat(s′, a′)
    q′ = min.(p.target_qnetwork1(q′_input), p.target_qnetwork2(q′_input))

    y = r .+ γ .* (1 .- t) .* vec(q′ .- α .* log_π)

    # Train Q Networks
    q_input = vcat(s, a)

    q_grad_1 = gradient(Flux.params(p.qnetwork1)) do
        q1 = p.qnetwork1(q_input) |> vec
        η * (0.5 * mse(q1, y) + α * CQL_loss(q1, y))
    end
    update!(p.qnetwork1, q_grad_1)
    q_grad_2 = gradient(Flux.params(p.qnetwork2)) do
        q2 = p.qnetwork1(q_input) |> vec
        η * (0.5 * mse(q2, y) + α * CQL_loss(q2, y))
    end
    update!(p.qnetwork2, q_grad_2)

    # Train Policy
    p_grad = gradient(Flux.params(p.policy)) do
        a, log_π = evaluate(p, s)
        q_input = vcat(s, a)
        q = min.(p.qnetwork1(q_input), p.qnetwork2(q_input))
        reward = mean(q)
        entropy = mean(log_π)
        Zygote.ignore() do 
            p.reward_term = reward
            p.entropy_term = entropy
        end
        ηₐ * (entropy - reward)
    end
    update!(p.policy, p_grad)

    # Tune entropy automatically
    if p.automatic_entropy_tuning
        p.α -= p.lr_alpha * mean(-log_π .- p.target_entropy)
    end

    # polyak averaging
    for (dest, src) in zip(
        Flux.params([p.target_qnetwork1, p.target_qnetwork2]),
        Flux.params([p.qnetwork1, p.qnetwork2]),
    )
        dest .= (1 - τ) .* dest .+ τ .* src
    end
end