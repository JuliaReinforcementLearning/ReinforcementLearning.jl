export FisherBRCLearner

mutable struct EntropyBC{A<:NeuralNetworkApproximator}
    policy::A
    α::Float32
    lr_alpha::Float32
    target_entropy::Float32
    # Logging
    policy_loss::Float32
end

mutable struct FisherBRCLearner{
    BA<:NeuralNetworkApproximator,
    BC1<:NeuralNetworkApproximator,
    BC2<:NeuralNetworkApproximator,
    R<:AbstractRNG,
} <: AbstractLearner
    policy::BA
    behavior_policy::EntropyBC
    qnetwork1::BC1
    qnetwork2::BC2
    target_qnetwork1::BC1
    target_qnetwork2::BC2
    γ::Float32
    τ::Float32
    α::Float32
    f_reg::Float32
    reward_bonus::Float32
    batch_size::Int
    pretrain_step::Int
    update_freq::Int
    update_step::Int
    lr_alpha::Float32
    target_entropy::Float32
    rng::R
    # Logging
    qnetwork_loss::Float32
    policy_loss::Float32
end

"""
    FisherBRCLearner(;kwargs...)

See paper: [Offline reinforcement learning with fisher divergence critic regularization](https://arxiv.org/abs/2103.08050).

# Keyword arguments

- `policy`, used to get action.
- `behavior_policy::EntropyBC`, used to estimate log μ(a|s).
- `qnetwork1`, used to get Q-values.
- `qnetwork2`, used to get Q-values.
- `target_qnetwork1`, used to estimate the target Q-values.
- `target_qnetwork2`, used to estimate the target Q-values.
- `γ::Float32 = 0.99f0`, reward discount rate.
- `τ::Float32 = 0.005f0`, the speed at which the target network is updated.
- `α::Float32 = 0.0f0`, entropy term.
- `f_reg::Float32 = 1.0f0`, the weight of gradient penalty regularizer.
- `reward_bonus::Float32 = 5.0f0`, add extra value to the reward.
- `batch_size::Int = 32`
- `pretrain_step::Int = 1000`, the number of pre-training rounds.
- `update_freq::Int = 50`, the frequency of updating the `approximator`.
- `lr_alpha::Float32 = 0.003f0`, learning rate of tuning entropy.
- `action_dims::Int = 0`, the dimensionality of the action.
- `update_step::Int = 0`
- `rng = Random.GLOBAL_RNG`

`policy` is expected to output a tuple `(μ, logσ)` of mean and
log standard deviations for the desired action distributions, this
can be implemented using a `GaussianNetwork` in a `NeuralNetworkApproximator`.
"""
function FisherBRCLearner(;
    policy,
    behavior_policy,
    qnetwork1,
    qnetwork2,
    target_qnetwork1,
    target_qnetwork2,
    γ = 0.99f0,
    τ = 0.005f0,
    α = 0.0f0,
    f_reg = 1.0f0,
    reward_bonus = 5.0f0,
    batch_size = 32,
    pretrain_step = 1000,
    update_freq = 50,
    lr_alpha = 0.003f0,
    behavior_lr_alpha = 0.001f0,
    action_dims = 0,
    update_step = 0,
    rng = Random.GLOBAL_RNG,
)
    copyto!(qnetwork1, target_qnetwork1)  # force sync
    copyto!(qnetwork2, target_qnetwork2)  # force sync
    entropy_behavior_policy = EntropyBC(behavior_policy, 0.0f0, behavior_lr_alpha, Float32(-action_dims), 0.0f0)
    FisherBRCLearner(
        policy,
        entropy_behavior_policy,
        qnetwork1,
        qnetwork2,
        target_qnetwork1,
        target_qnetwork2,
        γ,
        τ,
        α,
        f_reg,
        reward_bonus,
        batch_size,
        pretrain_step,
        update_freq,
        update_step,
        lr_alpha,
        Float32(-action_dims),
        rng,
        0f0,
        0f0,
    )
end

function (l::FisherBRCLearner)(env)
    D = device(l.policy)
    s = send_to_device(D, state(env))
    s = Flux.unsqueeze(s, ndims(s) + 1)
    action = dropdims(l.policy(l.rng, s; is_sampling=true), dims=2)
end

function RLBase.update!(l::FisherBRCLearner, batch::NamedTuple{SARTS})
    if l.update_step == 0
        update_behavior_policy!(l.behavior_policy, batch)
    else
        update_learner!(l, batch)
    end
end

function update_behavior_policy!(l::EntropyBC, batch::NamedTuple{SARTS})
    D = device(l.policy)
    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    # Update behavior policy with entropy
    gs = gradient(Flux.params(l.policy)) do
        log_π = l.policy.model(s, a)
        _, entropy = l.policy.model(s; is_sampling=true, is_return_log_prob=true)
        loss = mean(l.α .* entropy .- log_π)
        # Update entropy
        ignore() do
            l.policy_loss = loss
            l.α -= l.lr_alpha * mean(-entropy .- l.target_entropy)
        end
        loss
    end
    update!(l.policy, gs)
end

function update_learner!(l::FisherBRCLearner, batch::NamedTuple{SARTS})
    D = device(l.policy)
    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    r .+= l.reward_bonus
    γ, τ, α = l.γ, l.τ, l.α

    a′ = l.policy(l.rng, s′; is_sampling=true)
    q′_input = vcat(s′, a′)
    target_q′ = min.(l.target_qnetwork1(q′_input), l.target_qnetwork2(q′_input))

    y = r .+ γ .* (1 .- t) .* vec(target_q′)

    # Train Q Networks
    a = reshape(a, :, l.batch_size)
    q_input = vcat(s, a)
    log_μ = l.behavior_policy.policy.model(s, a) |> vec
    a_policy = l.policy(l.rng, s; is_sampling=true)

    q_grad_1 = gradient(Flux.params(l.qnetwork1)) do
        q1 = l.qnetwork1(q_input) |> vec
        q1_grad_norm = gradient(Flux.params([a_policy])) do 
            q1_reg = mean(l.qnetwork1(vcat(s, a_policy)))
        end
        reg = mean(q1_grad_norm[a_policy] .^ 2)
        loss = mse(q1 .+ log_μ, y) + l.f_reg * reg
        ignore() do 
            l.qnetwork_loss = loss
        end
        loss
    end
    update!(l.qnetwork1, q_grad_1)

    q_grad_2 = gradient(Flux.params(l.qnetwork2)) do
        q2 = l.qnetwork2(q_input) |> vec
        q2_grad_norm = gradient(Flux.params([a_policy])) do 
            q2_reg = mean(l.qnetwork2(vcat(s, a_policy)))
        end
        reg = mean(q2_grad_norm[a_policy] .^ 2)
        loss = mse(q2 .+ log_μ, y) + l.f_reg * reg
        ignore() do 
            l.qnetwork_loss += loss
        end
        loss
    end
    update!(l.qnetwork2, q_grad_2)

    # Train Policy
    p_grad = gradient(Flux.params(l.policy)) do
        a, log_π = l.policy(l.rng, s; is_sampling=true, is_return_log_prob=true)
        q_input = vcat(s, a)
        q = min.(l.qnetwork1(q_input), l.qnetwork2(q_input)) .+ log_μ
        policy_loss = mean(α .* log_π .- q)
        ignore() do
            l.policy_loss = policy_loss
            # Tune entropy automatically
            l.α -= l.lr_alpha * mean(-log_π .- l.target_entropy)
        end
        policy_loss
    end
    update!(l.policy, p_grad)

    # polyak averaging
    for (dest, src) in zip(
        Flux.params([l.target_qnetwork1, l.target_qnetwork2]),
        Flux.params([l.qnetwork1, l.qnetwork2]),
    )
        dest .= (1 - τ) .* dest .+ τ .* src
    end
end
