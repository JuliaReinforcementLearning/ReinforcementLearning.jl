export PLASLearner

mutable struct PLASLearner{
    BA1<:NeuralNetworkApproximator,
    BA2<:NeuralNetworkApproximator,
    BC1<:NeuralNetworkApproximator,
    BC2<:NeuralNetworkApproximator,
    V<:NeuralNetworkApproximator,
    R<:AbstractRNG,
} <: AbstractLearner
    policy::BA1
    target_policy::BA2
    qnetwork1::BC1
    qnetwork2::BC2
    target_qnetwork1::BC1
    target_qnetwork2::BC2
    vae::V
    γ::Float32
    τ::Float32
    λ::Float32
    batch_size::Int
    pretrain_step::Int
    update_freq::Int
    update_step::Int
    rng::R
    # Logging
    actor_loss::Float32
    critic_loss::Float32
end

"""
    PLASLearner(;kwargs...)

See [Latent Action Space for Offline Reinforcement Learning](https://arxiv.org/abs/2011.07213)

# Keyword arguments
- `policy`, used to get latent action.
- `target_policy`, similar to `policy`, but used to estimate the target.
- `qnetwork1`, used to get Q-values.
- `qnetwork2`, used to get Q-values.
- `target_qnetwork1`, used to estimate the target Q-values.
- `target_qnetwork2`, used to estimate the target Q-values.
- `vae`, used for mapping hidden actions to actions. This
can be implemented using a `VAE` in a `NeuralNetworkApproximator`.
- `γ::Float32 = 0.99f0`, reward discount rate.
- `τ::Float32 = 0.005f0`, the speed at which the target network is updated.
- `λ::Float32 = 0.75f0`, used for Clipped Double Q-learning.
- `batch_size::Int = 32`
- `pretrain_step::Int = 1000`, the number of pre-training rounds.
- `update_freq::Int = 50`, the frequency of updating the `approximator`.
- `update_step::Int = 0`
- `rng = Random.GLOBAL_RNG`
"""
function PLASLearner(;
    policy,
    target_policy,
    qnetwork1,
    qnetwork2,
    target_qnetwork1,
    target_qnetwork2,
    vae,
    γ = 0.99f0,
    τ = 0.005f0,
    λ = 0.75f0,
    batch_size = 32,
    pretrain_step = 10000,
    update_freq = 50,
    update_step = 0,
    rng = Random.GLOBAL_RNG,
)
    copyto!(policy, target_policy)  # force sync
    copyto!(qnetwork1, target_qnetwork1)  # force sync
    copyto!(qnetwork2, target_qnetwork2)  # force sync
    PLASLearner(
        policy,
        target_policy,
        qnetwork1,
        qnetwork2,
        target_qnetwork1,
        target_qnetwork2,
        vae,
        γ,
        τ,
        λ,
        batch_size,
        pretrain_step,
        update_freq,
        update_step,
        rng,
        0.0f0,
        0.0f0,
    )
end

function (l::PLASLearner)(env)
    s = send_to_device(device(l.policy), state(env))
    s = Flux.unsqueeze(s, ndims(s) + 1)
    latent_action = tanh.(l.policy(s))
    action = dropdims(decode(l.vae.model, s, latent_action), dims = 2)
end

function RLBase.update!(l::PLASLearner, batch::NamedTuple{SARTS})
    if l.update_step == 0
        update_vae!(l, batch)
    else
        update_learner!(l, batch)
    end
end

function update_vae!(l::PLASLearner, batch::NamedTuple{SARTS})
    s, a, r, t, s′ = send_to_device(device(l.vae), batch)
    a = reshape(a, :, l.batch_size)
    vae_grad = gradient(Flux.params(l.vae)) do
        recon_loss, kl_loss = vae_loss(l.vae.model, s, a)
        0.5f0 * kl_loss + recon_loss
    end
    update!(l.vae, vae_grad)
end

function update_learner!(l::PLASLearner, batch::NamedTuple{SARTS})
    s, a, r, t, s′ = send_to_device(device(l.qnetwork1), batch)

    γ, τ, λ = l.γ, l.τ, l.λ

    latent_action′ = tanh.(l.target_policy(s′))
    action′ = decode(l.vae.model, s′, latent_action′)
    q′_input = vcat(s′, action′)
    q′ =
        λ .* min.(l.target_qnetwork1(q′_input), l.target_qnetwork2(q′_input)) +
        (1 - λ) .* max.(l.target_qnetwork1(q′_input), l.target_qnetwork2(q′_input))

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

    # Train Policy
    p_grad = gradient(Flux.params(l.policy)) do
        latent_action = tanh.(l.policy(s))
        action = decode(l.vae.model, s, latent_action)
        actor_loss = -mean(l.qnetwork1(vcat(s, action)))
        ignore() do
            l.actor_loss = actor_loss
        end
        actor_loss
    end
    update!(l.policy, p_grad)

    # polyak averaging
    for (dest, src) in zip(
        Flux.params([l.target_policy, l.target_qnetwork1, l.target_qnetwork2]),
        Flux.params([l.policy, l.qnetwork1, l.qnetwork2]),
    )
        dest .= (1 - τ) .* dest .+ τ .* src
    end
end
