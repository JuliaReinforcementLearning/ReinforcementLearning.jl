export CRRLearner

"""
    CRRLearner(;kwargs)

See paper: [Critic Regularized Regression](https://arxiv.org/abs/2006.15134).

# Keyword arguments

- `approximator`::[`ActorCritic`](@ref): used to get Q-values (Critic) and logits (Actor) of a state.
- `target_approximator`::[`ActorCritic`](@ref): similar to `approximator`, but used to estimate the target.
- `γ::Float32`, reward discount rate.
- `batch_size::Int=32`
- `policy_improvement_mode::Symbol=:exp`, type of the weight function f. Possible values: :binary/:exp.
- `ratio_upper_bound::Float32`, when `policy_improvement_mode` is ":exp", the value of the exp function is upper-bounded by this parameter.
- `β::Float32`,  when `policy_improvement_mode` is ":exp", this is the denominator of the exp function.
- `advantage_estimator::Symbol=:mean`, type of the advantage estimate \\hat{A}. Possible values: :mean/:max.
- `m::Int=4`, when `continuous=true`, sample `m` action to estimate \\hat{A}.
- `update_freq::Int`: the frequency of updating the `approximator`.
- `update_step::Int=0`
- `target_update_freq::Int`: the frequency of syncing `target_approximator`.
- `continuous::Bool`: type of action space.
- `rng = Random.GLOBAL_RNG`
"""
mutable struct CRRLearner{
    Aq<:ActorCritic,
    At<:ActorCritic,
    R<:AbstractRNG,
} <: AbstractLearner
    approximator::Aq
    target_approximator::At
    γ::Float32
    batch_size::Int
    policy_improvement_mode::Symbol
    ratio_upper_bound::Float32
    β::Float32
    advantage_estimator::Symbol
    m::Int
    update_freq::Int
    update_step::Int
    target_update_freq::Int
    continuous::Bool
    rng::R
    # for logging
    actor_loss::Float32
    critic_loss::Float32
end

function CRRLearner(;
    approximator::Aq,
    target_approximator::At,
    γ::Float32 = 0.99f0,
    batch_size::Int = 32,
    policy_improvement_mode::Symbol = :exp,
    ratio_upper_bound::Float32 = 20.0f0,
    β::Float32 = 1.0f0,
    advantage_estimator::Symbol = :mean,
    m::Int = 4,
    update_freq::Int = 10,
    update_step::Int = 0,
    target_update_freq::Int = 100,
    continuous::Bool,
    rng = Random.GLOBAL_RNG,
) where {Aq<:ActorCritic, At<:ActorCritic}
    copyto!(approximator, target_approximator)
    CRRLearner(
        approximator,
        target_approximator,
        γ,
        batch_size,
        policy_improvement_mode,
        ratio_upper_bound,
        β,
        advantage_estimator,
        m,
        update_freq,
        update_step,
        target_update_freq,
        continuous,
        rng,
        0.0f0,
        0.0f0,
    )
end

Flux.functor(x::CRRLearner) = (Q = x.approximator, Qₜ = x.target_approximator),
y -> begin
    x = @set x.approximator = y.Q
    x = @set x.target_approximator = y.Qₜ
    x
end

function (learner::CRRLearner)(env)
    s = state(env)
    s = Flux.unsqueeze(s, ndims(s) + 1)
    s = send_to_device(device(learner), s)
    if learner.continuous
        learner.approximator.actor(s; is_sampling=true) |> vec |> send_to_host
    else
        learner.approximator.actor(s) |> vec |> send_to_host
    end
end

function RLBase.update!(learner::CRRLearner, batch::NamedTuple)
    if learner.continuous
        continuous_update!(learner, batch)
    else
        discrete_update!(learner, batch)
    end
end

function continuous_update!(learner::CRRLearner, batch::NamedTuple)
    AC = learner.approximator
    target_AC = learner.target_approximator
    γ = learner.γ
    β = learner.β
    batch_size = learner.batch_size
    policy_improvement_mode = learner.policy_improvement_mode
    ratio_upper_bound = learner.ratio_upper_bound
    advantage_estimator = learner.advantage_estimator
    D = device(AC)

    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    a = reshape(a, :, batch_size)
    r = reshape(r, :, batch_size)
    t = reshape(t, :, batch_size)

    target_a_t = target_AC.actor(s′; is_sampling=true)
    target_q_input = vcat(s′, target_a_t)
    expected_target_q = target_AC.critic(target_q_input)

    target = r .+ γ .* (1 .- t) .* expected_target_q

    q_t = send_to_device(D, Matrix{Float32}(undef, learner.m, batch_size))
    for i in 1:learner.m
        a_sample = AC.actor(s; is_sampling=true)
        q_t[i, :] = AC.critic(vcat(s, a_sample))
    end

    ps = Flux.params(AC)
    gs = gradient(ps) do
        # Critic loss
        qa_t = AC.critic(vcat(s, a))
        critic_loss = Flux.Losses.mse(qa_t, target)
        
        # Actor loss
        log_π = AC.actor(s, a)

        if advantage_estimator == :max
            advantage = qa_t .- maximum(q_t, dims=1)
        elseif advantage_estimator == :mean
            advantage = qa_t .- mean(q_t, dims=1)
        else
            error("Wrong parameter.")
        end

        if policy_improvement_mode == :binary
            actor_loss_coef = (advantage .> 0.0f0)
        elseif policy_improvement_mode == :exp
            actor_loss_coef = clamp.(exp.(advantage ./ β), 0, ratio_upper_bound)
        else
            error("Wrong parameter.")
        end

        actor_loss = mean(-log_π .* actor_loss_coef)

        ignore() do
            learner.actor_loss = actor_loss
            learner.critic_loss = critic_loss
        end
        
        actor_loss + critic_loss
    end

    update!(AC, gs)
end

function discrete_update!(learner::CRRLearner, batch::NamedTuple)
    AC = learner.approximator
    target_AC = learner.target_approximator
    γ = learner.γ
    β = learner.β
    batch_size = learner.batch_size
    policy_improvement_mode = learner.policy_improvement_mode
    ratio_upper_bound = learner.ratio_upper_bound
    advantage_estimator = learner.advantage_estimator
    D = device(AC)

    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    a = CartesianIndex.(a, 1:batch_size)
    r = send_to_device(D, reshape(r, :, batch_size))
    t = send_to_device(D, reshape(t, :, batch_size))

    target_a_t = softmax(target_AC.actor(s′))
    target_q_t = target_AC.critic(s′)
    expected_target_q = sum(target_a_t .* target_q_t, dims=1)

    target = r .+ γ .* (1 .- t) .* expected_target_q

    ps = Flux.params(AC)
    gs = gradient(ps) do
        # Critic loss
        q_t = AC.critic(s)
        qa_t = reshape(q_t[a], :, batch_size)
        critic_loss = Flux.Losses.mse(qa_t, target)
        
        # Actor loss
        a_t = softmax(AC.actor(s))

        if advantage_estimator == :max
            advantage = qa_t .- maximum(q_t, dims=1)
        elseif advantage_estimator == :mean
            advantage = qa_t .- mean(q_t, dims=1)
        else
            error("Wrong parameter.")
        end

        if policy_improvement_mode == :binary
            actor_loss_coef = (advantage .> 0.0f0)
        elseif policy_improvement_mode == :exp
            actor_loss_coef = clamp.(exp.(advantage ./ β), 0, ratio_upper_bound)
        else
            error("Wrong parameter.")
        end
        
        actor_loss = mean(-log.(a_t[a]) .* actor_loss_coef)

        ignore() do
            learner.actor_loss = actor_loss
            learner.critic_loss = critic_loss
        end
        
        actor_loss + critic_loss
    end

    update!(AC, gs)
end