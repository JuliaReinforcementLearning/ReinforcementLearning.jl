export BEARLearner

mutable struct BEARLearner{
    BA1<:NeuralNetworkApproximator,
    BA2<:NeuralNetworkApproximator,
    BC1<:NeuralNetworkApproximator,
    BC2<:NeuralNetworkApproximator,
    V<:NeuralNetworkApproximator,
    L<:NeuralNetworkApproximator,
    R<:AbstractRNG,
} <: AbstractLearner
    policy::BA1
    target_policy::BA2
    qnetwork1::BC1
    qnetwork2::BC2
    target_qnetwork1::BC1
    target_qnetwork2::BC2
    vae::V
    log_α::L
    γ::Float32
    τ::Float32
    λ::Float32
    ε::Float32
    p::Int
    max_log_α::Float32
    min_log_α::Float32
    sample_num::Int
    kernel_type::Symbol
    mmd_σ::Float32
    batch_size::Int
    update_freq::Int
    update_step::Int
    rng::R
    # Logging
    actor_loss::Float32
    critic_loss::Float32
    mmd_loss
end

"""
    BEARLearner(;kwargs...)

See [Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction](https://arxiv.org/abs/1906.00949). This implmentation refers to the [official python code](https://github.com/aviralkumar2907/BEAR).

# Keyword arguments
- `policy`, used to get latent action.
- `target_policy`, similar to `policy`, but used to estimate the target.
- `qnetwork1`, used to get Q-values.
- `qnetwork2`, used to get Q-values.
- `target_qnetwork1`, used to estimate the target Q-values.
- `target_qnetwork2`, used to estimate the target Q-values.
- `vae`, used for sampling action to calculate MMD loss. This
can be implemented using a `VAE` in a `NeuralNetworkApproximator`.
- `log_α`, lagrange multiplier implemented by a `NeuralNetworkApproximator`.
- `γ::Float32 = 0.99f0`, reward discount rate.
- `τ::Float32 = 0.005f0`, the speed at which the target network is updated.
- `λ::Float32 = 0.75f0`, used for Clipped Double Q-learning.
- `ε::Float32 = 0.05f0`, threshold of MMD loss.
- `p::Int = 10`, the number of state-action pairs used when calculating the Q value.
- `max_log_α::Float32 = 10.0f0`, maximum value of `log_α`.
- `min_log_α::Float32 = 10.0f0`, minimum value of `log_α`.
- `sample_num::Int = 10`, the number of sample action to calculate MMD loss.
- `kernel_type::Symbol = :laplacian`, the method of calculating MMD loss. Possible values: :laplacian/:gaussian.
- `mmd_σ::Float32 = 10.0f0`, the parameter used for calculating MMD loss.
- `batch_size::Int = 32`
- `update_freq::Int = 50`, the frequency of updating the `approximator`.
- `update_step::Int = 0`
- `rng = Random.GLOBAL_RNG`
"""
function BEARLearner(;
    policy,
    target_policy,
    qnetwork1,
    qnetwork2,
    target_qnetwork1,
    target_qnetwork2,
    vae,
    log_α,
    γ = 0.99f0,
    τ = 0.005f0,
    λ = 0.75f0,
    ε = 0.05f0,
    p = 10,
    max_log_α = 10.0f0,
    min_log_α = -5.0f0,
    sample_num = 10,
    kernel_type = :laplacian,
    mmd_σ = 10.0f0,
    batch_size = 32,
    update_freq = 50,
    update_step = 0,
    rng = Random.GLOBAL_RNG,
)
    copyto!(policy, target_policy)  # force sync
    copyto!(qnetwork1, target_qnetwork1)  # force sync
    copyto!(qnetwork2, target_qnetwork2)  # force sync
    BEARLearner(
        policy,
        target_policy,
        qnetwork1,
        qnetwork2,
        target_qnetwork1,
        target_qnetwork2,
        vae,
        log_α,
        γ,
        τ,
        λ,
        ε,
        p,
        max_log_α,
        min_log_α,
        sample_num,
        kernel_type,
        mmd_σ,
        batch_size,
        update_freq,
        update_step,
        rng,
        0.0f0,
        0.0f0,
        0.0f0,
    )
end

function (l::BEARLearner)(env)
    s = send_to_device(device(l.policy), state(env))
    s = Flux.unsqueeze(s, ndims(s) + 1)
    s = repeat(s, outer=(1, 1, l.p))
    action = l.policy(l.rng, s; is_sampling=true)
    q_value = l.qnetwork1(vcat(s, action))
    idx = argmax(q_value)
    action[idx]
end

function RLBase.update!(l::BEARLearner, batch::NamedTuple{SARTS})
    D = device(l.qnetwork1)
    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    γ, τ, λ = l.γ, l.τ, l.λ

    update_vae!(l, s, a)
    
    repeat_s′ = repeat(s′, outer=(1, 1, l.p))
    repeat_action′ = l.target_policy(l.rng, repeat_s′, is_sampling=true)

    q′_input = vcat(repeat_s′, repeat_action′)

    q′ = maximum(λ .* min.(l.target_qnetwork1(q′_input), l.target_qnetwork2(q′_input)) + (1 - λ) .* max.(l.target_qnetwork1(q′_input), l.target_qnetwork2(q′_input)), dims=3)

    y = r .+ γ .* (1 .- t) .* vec(q′)

    # Train Q Networks
    a = reshape(a, :, l.batch_size)
    q_input = vcat(s, a)

    q_grad_1 = gradient(Flux.params(l.qnetwork1)) do
        q1 = l.qnetwork1(q_input) |> vec
        loss = mse(q1, y)
        ignore() do 
            l.critic_loss = loss
        end
        loss
    end
    update!(l.qnetwork1, q_grad_1)

    q_grad_2 = gradient(Flux.params(l.qnetwork2)) do
        q2 = l.qnetwork2(q_input) |> vec
        loss = mse(q2, y)
        ignore() do 
            l.critic_loss += loss
        end
        loss
    end
    update!(l.qnetwork2, q_grad_2)

    repeat_s = repeat(s, outer=(1, 1, l.p))
    repeat_a = repeat(a, outer=(1, 1, l.p))
    repeat_q1 = mean(l.target_qnetwork1(vcat(repeat_s, repeat_a)), dims=(1, 3))
    repeat_q2 = mean(l.target_qnetwork2(vcat(repeat_s, repeat_a)), dims=(1, 3))
    q = vec(min.(repeat_q1, repeat_q2))

    alpha = exp(l.log_α.model[1])

    # Train Policy
    p_grad = gradient(Flux.params(l.policy)) do
        raw_sample_action = decode(l.vae.model, repeat(s, outer=(1, 1, l.sample_num)); is_normalize=false)  # action_dim * batch_size * sample_num
        raw_actor_action = l.policy(repeat(s, outer=(1, 1, l.sample_num)); is_sampling=true) # action_dim * batch_size * sample_num

        mmd_loss = maximum_mean_discrepancy_loss(raw_sample_action, raw_actor_action, l.kernel_type, l.mmd_σ)

        actor_loss = mean(-q .+ alpha .* mmd_loss)
        ignore() do 
            l.actor_loss = actor_loss
            l.mmd_loss = mmd_loss
        end
        actor_loss
    end
    update!(l.policy, p_grad)

    # Update lagrange multiplier
    l_grad = gradient(Flux.params(l.log_α)) do 
        mean(-q .+ alpha .* (l.mmd_loss .- l.ε))
    end
    update!(l.log_α, l_grad)
    
    clamp!(l.log_α.model, l.min_log_α, l.max_log_α)
    
    # polyak averaging
    for (dest, src) in zip(
        Flux.params([l.target_policy, l.target_qnetwork1, l.target_qnetwork2]),
        Flux.params([l.policy, l.qnetwork1, l.qnetwork2]),
    )
        dest .= (1 - τ) .* dest .+ τ .* src
    end
end

function update_vae!(l::BEARLearner, s, a)
    a = reshape(a, :, l.batch_size)
    vae_grad = gradient(Flux.params(l.vae)) do
        recon_loss, kl_loss = vae_loss(l.vae.model, s, a)
        0.5f0 * kl_loss + recon_loss
    end
    update!(l.vae, vae_grad)
end
