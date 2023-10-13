export SACPolicy

mutable struct SACPolicy{
    BA<:Approximator{<:SoftGaussianNetwork},
    BC1<:TargetNetwork,
    BC2<:TargetNetwork,
    P,
    R<:AbstractRNG,
    DR<:AbstractRNG,
} <: AbstractPolicy
    policy::BA
    qnetwork1::BC1
    qnetwork2::BC2
    γ::Float32
    α::Float32
    start_steps::Int
    start_policy::P
    automatic_entropy_tuning::Bool
    lr_alpha::Float32
    target_entropy::Float32
    update_step::Int
    rng::R
    device_rng::DR
    # Logging
    reward_term::Float32
    entropy_term::Float32
end

"""
    SACPolicy(;kwargs...)

# Keyword arguments

- `policy`, used to get action.
- `qnetwork1::TargetNetwork`, used to get Q-values.
- `qnetwork2::TargetNetwork`, used to get Q-values.
- `start_policy`, 
- `γ::Float32 = 0.99f0`, reward discount rate.
- `α::Float32 = 0.2f0`, entropy term.
- `start_steps = 10000`, number of steps where start_policy is used to sample actions
- `update_after = 1000`, number of steps before starting to update policy
- `automatic_entropy_tuning::Bool = true`, whether to automatically tune the entropy.
- `lr_alpha::Float32 = 0.003f0`, learning rate of tuning entropy.
- `action_dims = 0`, the dimensionality of the action. if `automatic_entropy_tuning = true`, must enter this parameter.
- `update_step = 0`,
- `rng = Random.default_rng()`, used to sample batch from trajectory or action from action distribution.
- `device_rng = Random.default_rng()`, should be set to `CUDA.CURAND.RNG()` if the `policy` is set to work with `CUDA.jl`

`policy` is expected to output a tuple `(μ, σ)` of mean and
standard deviations for the desired action distributions, this
can be implemented using a `SoftGaussianNetwork` in a `Approximator`.

Implemented based on http://arxiv.org/abs/1812.05905
"""
function SACPolicy(;
    policy,
    qnetwork1,
    qnetwork2,
    γ=0.99f0,
    α=0.2f0,
    start_steps=10000,
    automatic_entropy_tuning=true,
    lr_alpha=0.003f0,
    action_dims=0,
    update_step=0,
    start_policy=update_step == 0 ? identity : policy,
    rng=Random.default_rng(),
    device_rng=Random.default_rng()
)
    if automatic_entropy_tuning
        @assert action_dims != 0
    end
    SACPolicy(
        policy,
        qnetwork1,
        qnetwork2,
        γ,
        α,
        start_steps,
        start_policy,
        automatic_entropy_tuning,
        lr_alpha,
        Float32(-action_dims),
        update_step,
        rng,
        device_rng,
        0.0f0,
        0.0f0,
    )
end

# TODO: handle Training/Testing mode
function RLBase.plan!(p::SACPolicy, env)
    p.update_step += 1
    if p.update_step <= p.start_steps
        action = RLBase.plan!(p.start_policy, env)
        if(size(action[1]) != ())
            action = reduce(hcat, action)
        end
        action
    else
        D = device(p.policy)
        s = send_to_device(D, state(env))
        s = Flux.unsqueeze(s, dims=ndims(s) + 1)
        # trainmode:
        action = RLCore.forward(p.policy, p.device_rng, s; is_sampling=true)
        action = dropdims(action, dims=ndims(action)) # Single action vec, drop second dim
        send_to_host(action)

        # testmode:
        # if testing dont sample an action, but act deterministically by
        # taking the "mean" action
        # action = dropdims(p.policy(s)[1], dims=2) 
    end
end

function RLBase.optimise!(
    p::SACPolicy,
    ::PostActStage,
    traj::Trajectory
)
    for batch in traj
        update_critic!(p, batch)
        update_actor!(p, batch)
    end
end

function soft_q_learning_target(p::SACPolicy, r, t, s′, α, γ)
    a′, log_π = RLCore.forward(p.policy,p.device_rng, s′; is_sampling=true, is_return_log_prob=true)
    q′_input = vcat(s′, a′)
    q′ = min.(target(p.qnetwork1)(q′_input), target(p.qnetwork2)(q′_input))

    r .+ γ .* (1 .- t) .* dropdims(q′ .- α .* log_π, dims=1)
end

function q_learning_loss(qnetwork, s, a, y)
    q_input = vcat(s, a)
    q = dropdims(model(qnetwork)(q_input), dims=1)
    mse(q, y)
end

function update_critic!(p::SACPolicy, batch::NamedTuple{SS′ART})
    s, s′, a, r, t = send_to_device(device(p.qnetwork1), batch)

    γ, α = p.γ, p.α

    y = soft_q_learning_target(p, r, t, s′, α, γ)


    # Train Q Networks
    q_grad_1 = gradient(Flux.params(model(p.qnetwork1))) do
        q_learning_loss(p.qnetwork1, s, a, y)
    end
    RLBase.optimise!(p.qnetwork1, q_grad_1)

    q_grad_2 = gradient(Flux.params(model(p.qnetwork2))) do
        q_learning_loss(p.qnetwork2, s, a, y)
    end
    RLBase.optimise!(p.qnetwork2, q_grad_2)
end

function update_actor!(p::SACPolicy, batch::NamedTuple{SS′ART})
    s, s′, a, r, t = send_to_device(device(p.qnetwork1), batch)

    # Train Policy
    p_grad = gradient(Flux.params(p.policy)) do
        a, log_π = RLCore.forward(p.policy, p.device_rng, s; is_sampling=true, is_return_log_prob=true)
        q_input = vcat(s, a)
        q = min.(model(p.qnetwork1)(q_input), model(p.qnetwork2)(q_input))
        reward = mean(q)
        entropy = mean(log_π)
        ignore_derivatives() do
            p.reward_term = reward
            p.entropy_term = entropy
        end
        p.α * entropy - reward
    end
    RLBase.optimise!(p.policy, p_grad)

    # Tune entropy automatically
    if p.automatic_entropy_tuning
        p.α -= p.lr_alpha * mean(-log_π .- p.target_entropy)
    end
end
