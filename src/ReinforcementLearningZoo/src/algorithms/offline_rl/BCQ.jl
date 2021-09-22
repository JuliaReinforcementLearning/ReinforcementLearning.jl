export BCQLearner

mutable struct BCQLearner{
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
    p::Int
    batch_size::Int
    start_step::Int
    update_freq::Int
    update_step::Int
    rng::R
    # Logging
    actor_loss::Float32
    critic_loss::Float32
end

"""
    BCQLearner(;kwargs...)

See [Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/abs/1812.02900)

# Keyword arguments
- `policy`, used to get action with perturbation. This can be implemented using a `PerturbationNetwork` in a `NeuralNetworkApproximator`.
- `target_policy`, similar to `policy`, but used to estimate the target. This can be implemented using a `PerturbationNetwork` in a `NeuralNetworkApproximator`.
- `qnetwork1`, used to get Q-values.
- `qnetwork2`, used to get Q-values.
- `target_qnetwork1`, used to estimate the target Q-values.
- `target_qnetwork2`, used to estimate the target Q-values.
- `vae`, used for sampling action. This
can be implemented using a `VAE` in a `NeuralNetworkApproximator`.
- `γ::Float32 = 0.99f0`, reward discount rate.
- `τ::Float32 = 0.005f0`, the speed at which the target network is updated.
- `λ::Float32 = 0.75f0`, used for Clipped Double Q-learning.
- `p::Int = 10`, the number of state-action pairs used when calculating the Q value.
- `batch_size::Int = 32`
- `start_step::Int = 1000`
- `update_freq::Int = 50`, the frequency of updating the `approximator`.
- `update_step::Int = 0`
- `rng = Random.GLOBAL_RNG`
"""
function BCQLearner(;
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
    p = 10,
    batch_size = 32,
    start_step = 1000,
    update_freq = 50,
    update_step = 0,
    rng = Random.GLOBAL_RNG,
)
    copyto!(policy, target_policy)  # force sync
    copyto!(qnetwork1, target_qnetwork1)  # force sync
    copyto!(qnetwork2, target_qnetwork2)  # force sync
    BCQLearner(
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
        p,
        batch_size,
        start_step,
        update_freq,
        update_step,
        rng,
        0.0f0,
        0.0f0,
    )
end

function (l::BCQLearner)(env)
    s = send_to_device(device(l.policy), state(env))
    s = Flux.unsqueeze(s, ndims(s) + 1)
    s = repeat(s, outer=(1, 1, l.p))
    action = l.policy(s, decode(l.vae.model, s))
    q_value = l.qnetwork1(vcat(s, action))
    idx = argmax(q_value)
    action[idx]
end

function RLBase.update!(l::BCQLearner, batch::NamedTuple{SARTS})
    update_vae!(l, batch)
    if l.update_step >= l.start_step
        update_learner!(l, batch)
    end
end

function update_vae!(l::BCQLearner, batch::NamedTuple{SARTS})
    D = device(l.vae)
    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    a = reshape(a, :, l.batch_size)
    vae_grad = gradient(Flux.params(l.vae)) do
        recon_loss, kl_loss = vae_loss(l.vae.model, s, a)
        0.5f0 * kl_loss + recon_loss
    end
    update!(l.vae, vae_grad)
end

function update_learner!(l::BCQLearner, batch::NamedTuple{SARTS})
    D = device(l.qnetwork1)
    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)

    γ, τ, λ = l.γ, l.τ, l.λ

    repeat_s′ = repeat(s′, outer=(1, 1, l.p))
    repeat_a′ = l.target_policy(repeat_s′, decode(l.vae.model, repeat_s′))

    q′_input = vcat(repeat_s′, repeat_a′)
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

    # Train Policy
    p_grad = gradient(Flux.params(l.policy)) do
        sampled_action = decode(l.vae.model, s)
        perturbed_action = l.policy(s, sampled_action)
        actor_loss = -mean(l.qnetwork1(vcat(s, perturbed_action)))
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
