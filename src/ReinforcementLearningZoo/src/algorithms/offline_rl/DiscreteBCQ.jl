export BCQDLearner

"""
    BCQDLearner(;kwargs)

See paper: [Benchmarking Batch Deep Reinforcement Learning Algorithms](https://arxiv.org/abs/1910.01708).

# Keyword arguments

- `approximator`::[`ActorCritic`](@ref): used to get Q-values (Critic) and logits (Actor) of a state.
- `target_approximator`::[`ActorCritic`](@ref): similar to `approximator`, but used to estimate the target.
- `γ::Float32 = 0.99f0`, reward discount rate.
- `τ::Float32 = 0.005f0`, the speed at which the target network is updated.
- `θ::Float32 = 0.99f0`, regularization coefficient.
- `threshold::Float32 = 0.3f0`, determine whether the action can be used to calculate the Q value.
- `batch_size::Int=32`
- `update_freq::Int`: the frequency of updating the `approximator`.
- `update_step::Int = 0`
- `rng = Random.GLOBAL_RNG`
"""
mutable struct BCQDLearner{
    Aq<:ActorCritic,
    At<:ActorCritic,
    R<:AbstractRNG,
} <: AbstractLearner
    approximator::Aq
    target_approximator::At
    γ::Float32
    τ::Float32
    θ::Float32
    threshold::Float32
    batch_size::Int
    update_freq::Int
    update_step::Int
    rng::R
    # for logging
    actor_loss::Float32
    critic_loss::Float32
end

function BCQDLearner(;
    approximator::Aq,
    target_approximator::At,
    γ::Float32 = 0.99f0,
    τ::Float32 = 0.005f0,
    θ::Float32 = 1f-2,
    threshold::Float32 = 0.3f0,
    batch_size::Int = 32,
    update_freq::Int = 10,
    update_step::Int = 0,
    rng = Random.GLOBAL_RNG,
) where {Aq<:ActorCritic, At<:ActorCritic}
    copyto!(approximator, target_approximator)
    BCQDLearner(
        approximator,
        target_approximator,
        γ,
        τ,
        θ,
        threshold,
        batch_size,
        update_freq,
        update_step,
        rng,
        0.0f0,
        0.0f0,
    )
end

Flux.functor(x::BCQDLearner) = (Q = x.approximator, Qₜ = x.target_approximator),
y -> begin
    x = @set x.approximator = y.Q
    x = @set x.target_approximator = y.Qₜ
    x
end

function (learner::BCQDLearner)(env)
    s = state(env)
    s = Flux.unsqueeze(s, ndims(s) + 1)
    s = send_to_device(device(learner), s)
    q = learner.approximator.critic(s)
    prob = softmax(learner.approximator.actor(s), dims=1)
    mask = Float32.((prob ./ maximum(prob, dims=1)) .> learner.threshold)
    new_q = q .* mask .+ (1.0f0 .- mask) .* -1f8
    new_q |> vec |> send_to_host
end

function RLBase.update!(learner::BCQDLearner, batch::NamedTuple)
    AC = learner.approximator
    target_AC = learner.target_approximator
    γ, τ, θ = learner.γ, learner.τ, learner.θ
    batch_size = learner.batch_size
    D = device(AC)

    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    a = CartesianIndex.(a, 1:batch_size)

    prob = softmax(AC.actor(s′))
    mask = Float32.((prob ./ maximum(prob, dims=1)) .> learner.threshold)
    q′ = AC.critic(s′)
    a′ = argmax(q′ .* mask .+ (1.0f0 .- mask) .* -1f8, dims=1)
    target_q = target_AC.critic(s′)

    target = r .+ γ .* (1 .- t) .* vec(target_q[a′])

    ps = Flux.params(AC)
    gs = gradient(ps) do
        # Critic loss
        q_t = AC.critic(s)
        critic_loss = Flux.Losses.huber_loss(q_t[a], target)
        
        # Actor loss
        logit = AC.actor(s)
        log_prob = -log.(softmax(logit, dims=1))
        actor_loss = mean(log_prob[a])

        ignore() do
            learner.actor_loss = actor_loss
            learner.critic_loss = critic_loss
        end
        
        actor_loss + critic_loss + θ * mean(logit .^ 2)
    end

    update!(AC, gs)

    # polyak averaging
    for (dest, src) in zip(
        Flux.params([learner.target_approximator]),
        Flux.params([learner.approximator]),
    )
        dest .= (1 - τ) .* dest .+ τ .* src
    end
end
