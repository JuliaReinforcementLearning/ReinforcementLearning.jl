export SACPolicy

mutable struct SACPolicy{
    BA<:NeuralNetworkApproximator,
    BC1<:NeuralNetworkApproximator,
    BC2<:NeuralNetworkApproximator,
    P,
    R<:AbstractRNG,
    DR<:AbstractRNG,
} <: AbstractPolicy
    policy::BA
    qnetwork1::BC1
    qnetwork2::BC2
    target_qnetwork1::BC1
    target_qnetwork2::BC2
    γ::Float32
    τ::Float32
    α::Float32
    batch_size::Int
    start_steps::Int
    start_policy::P
    update_after::Int
    update_freq::Int
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
- `qnetwork1`, used to get Q-values.
- `qnetwork2`, used to get Q-values.
- `target_qnetwork1 = deepcopy(qnetwork1)`, used to estimate the target Q-values.
- `target_qnetwork2 = deepcopy(qnetwork2)`, used to estimate the target Q-values.
- `start_policy`, 
- `γ::Float32 = 0.99f0`, reward discount rate.
- `τ::Float32 = 0.005f0`, the speed at which the target network is updated.
- `α::Float32 = 0.2f0`, entropy term.
- `batch_size = 32`,
- `start_steps = 10000`, number of steps where start_policy is used to sample actions
- `update_after = 1000`, number of steps before starting to update policy
- `update_freq = 50`, number of steps between each update
- `automatic_entropy_tuning::Bool = true`, whether to automatically tune the entropy.
- `lr_alpha::Float32 = 0.003f0`, learning rate of tuning entropy.
- `action_dims = 0`, the dimensionality of the action. if `automatic_entropy_tuning = true`, must enter this parameter.
- `update_step = 0`,
- `rng = Random.GLOBAL_RNG`, used to sample batch from trajectory or action from action distribution.
- `device_rng = Random.GLOBAL_RNG`, should be set to `CUDA.CURAND.RNG()` if the `policy` is set to work with `CUDA.jl`

`policy` is expected to output a tuple `(μ, logσ)` of mean and
log standard deviations for the desired action distributions, this
can be implemented using a `GaussianNetwork` in a `NeuralNetworkApproximator`.

Implemented based on http://arxiv.org/abs/1812.05905
"""
function SACPolicy(;
    policy,
    qnetwork1,
    qnetwork2,
    target_qnetwork1=deepcopy(qnetwork1),
    target_qnetwork2=deepcopy(qnetwork2),
    γ=0.99f0,
    τ=0.005f0,
    α=0.2f0,
    batch_size=32,
    start_steps=10000,
    update_after=1000,
    update_freq=50,
    automatic_entropy_tuning=true,
    lr_alpha=0.003f0,
    action_dims=0,
    update_step=0,
    start_policy=update_step == 0 ? identity : policy,
    rng=Random.GLOBAL_RNG,
    device_rng=Random.GLOBAL_RNG
)
    copyto!(qnetwork1, target_qnetwork1)  # force sync
    copyto!(qnetwork2, target_qnetwork2)  # force sync
    if automatic_entropy_tuning
        @assert action_dims != 0
    end
    SACPolicy(
        policy,
        qnetwork1,
        qnetwork2,
        target_qnetwork1,
        target_qnetwork2,
        γ,
        τ,
        α,
        batch_size,
        start_steps,
        start_policy,
        update_after,
        update_freq,
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
function (p::SACPolicy)(env)
    p.update_step += 1

    if p.update_step <= p.start_steps
        p.start_policy(env)
    else
        D = device(p.policy)
        s = send_to_device(D, state(env))
        s = Flux.unsqueeze(s, ndims(s) + 1)
        # trainmode:
        action = dropdims(p.policy(p.device_rng, s; is_sampling=true), dims=2) # Single action vec, drop second dim
        send_to_host(action)

        # testmode:
        # if testing dont sample an action, but act deterministically by
        # taking the "mean" action
        # action = dropdims(p.policy(s)[1], dims=2) 
    end
end

function RLBase.update!(
    p::SACPolicy,
    traj::CircularArraySARTTrajectory,
    ::AbstractEnv,
    ::PreActStage,
)
    length(traj) > p.update_after || return
    p.update_step % p.update_freq == 0 || return
    inds, batch = sample(p.rng, traj, BatchSampler{SARTS}(p.batch_size))
    update!(p, batch)
end

function RLBase.update!(p::SACPolicy, batch::NamedTuple{SARTS})
    s, a, r, t, s′ = send_to_device(device(p.qnetwork1), batch)

    γ, τ, α = p.γ, p.τ, p.α

    a′, log_π = p.policy(p.device_rng, s′; is_sampling=true, is_return_log_prob=true)
    q′_input = vcat(s′, a′)
    q′ = min.(p.target_qnetwork1(q′_input), p.target_qnetwork2(q′_input))

    y = r .+ γ .* (1 .- t) .* vec(q′ .- α .* log_π)

    # Train Q Networks
    q_input = vcat(s, a)

    q_grad_1 = gradient(Flux.params(p.qnetwork1)) do
        q1 = p.qnetwork1(q_input) |> vec
        mse(q1, y)
    end
    update!(p.qnetwork1, q_grad_1)
    q_grad_2 = gradient(Flux.params(p.qnetwork2)) do
        q2 = p.qnetwork2(q_input) |> vec
        mse(q2, y)
    end
    update!(p.qnetwork2, q_grad_2)

    # Train Policy
    p_grad = gradient(Flux.params(p.policy)) do
        a, log_π = p.policy(p.device_rng, s; is_sampling=true, is_return_log_prob=true)
        q_input = vcat(s, a)
        q = min.(p.qnetwork1(q_input), p.qnetwork2(q_input))
        reward = mean(q)
        entropy = mean(log_π)
        ignore() do
            p.reward_term = reward
            p.entropy_term = entropy
        end
        α * entropy - reward
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
