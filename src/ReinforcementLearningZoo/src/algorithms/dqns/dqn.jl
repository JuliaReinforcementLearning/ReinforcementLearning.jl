export DQNLearner

mutable struct DQNLearner{
    Tq<:AbstractApproximator,
    Tt<:AbstractApproximator,
    Tf,
    R<:AbstractRNG,
} <: AbstractLearner
    approximator::Tq
    target_approximator::Tt
    loss_func::Tf
    min_replay_history::Int
    update_freq::Int
    update_step::Int
    target_update_freq::Int
    sampler::NStepBatchSampler
    rng::R
    # for logging
    loss::Float32
    is_enable_double_DQN::Bool
end

"""
    DQNLearner(;kwargs...)

See paper: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

# Keywords

- `approximator`::[`AbstractApproximator`](@ref): used to get Q-values of a state.
- `target_approximator`::[`AbstractApproximator`](@ref): similar to `approximator`, but used to estimate the target (the next state).
- `loss_func`: the loss function.
- `γ::Float32=0.99f0`: discount rate.
- `batch_size::Int=32`
- `update_horizon::Int=1`: length of update ('n' in n-step update).
- `min_replay_history::Int=32`: number of transitions that should be experienced before updating the `approximator`.
- `update_freq::Int=4`: the frequency of updating the `approximator`.
- `target_update_freq::Int=100`: the frequency of syncing `target_approximator`.
- `stack_size::Union{Int, Nothing}=4`: use the recent `stack_size` frames to form a stacked state.
- `traces = SARTS`: set to `SLARTSL` if you are to apply to an environment of `FULL_ACTION_SET`.
- `rng = Random.GLOBAL_RNG`
- `is_enable_double_DQN = Bool`: enable double dqn, enabled by default.
"""
function DQNLearner(;
    approximator::Tq,
    target_approximator::Tt,
    loss_func::Tf,
    stack_size::Union{Int,Nothing} = nothing,
    γ::Float32 = 0.99f0,
    batch_size::Int = 32,
    update_horizon::Int = 1,
    min_replay_history::Int = 32,
    update_freq::Int = 1,
    target_update_freq::Int = 100,
    traces = SARTS,
    update_step = 0,
    rng = Random.GLOBAL_RNG,
    is_enable_double_DQN::Bool = true,
) where {Tq,Tt,Tf}
    copyto!(approximator, target_approximator)
    sampler = NStepBatchSampler{traces}(;
        γ = γ,
        n = update_horizon,
        stack_size = stack_size,
        batch_size = batch_size,
    )
    DQNLearner(
        approximator,
        target_approximator,
        loss_func,
        min_replay_history,
        update_freq,
        update_step,
        target_update_freq,
        sampler,
        rng,
        0.0f0,
        is_enable_double_DQN,
    )
end


Flux.functor(x::DQNLearner) = (Q = x.approximator, Qₜ = x.target_approximator),
y -> begin
    x = @set x.approximator = y.Q
    x = @set x.target_approximator = y.Qₜ
    x
end

function (learner::DQNLearner)(env)
    env |>
    state |>
    x ->
        Flux.unsqueeze(x, ndims(x) + 1) |>
        x ->
            send_to_device(device(learner), x) |>
            learner.approximator |>
            vec |>
            send_to_host
end

function RLBase.update!(learner::DQNLearner, batch::NamedTuple)
    Q = learner.approximator
    Qₜ = learner.target_approximator
    γ = learner.sampler.γ
    loss_func = learner.loss_func
    n = learner.sampler.n
    batch_size = learner.sampler.batch_size
    is_enable_double_DQN = learner.is_enable_double_DQN
    D = device(Q)

    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    a = CartesianIndex.(a, 1:batch_size)

    if is_enable_double_DQN
        q_values = Q(s′)
    else
        q_values = Qₜ(s′)
    end

    if haskey(batch, :next_legal_actions_mask)
        l′ = send_to_device(D, batch[:next_legal_actions_mask])
        q_values .+= ifelse.(l′, 0.0f0, typemin(Float32))
    end

    if is_enable_double_DQN
        selected_actions = dropdims(argmax(q_values, dims = 1), dims = 1)
        q′ = Qₜ(s′)[selected_actions]
    else
        q′ = dropdims(maximum(q_values; dims = 1), dims = 1)
    end

    G = r .+ γ^n .* (1 .- t) .* q′

    gs = gradient(params(Q)) do
        q = Q(s)[a]
        loss = loss_func(G, q)
        ignore() do
            learner.loss = loss
        end
        loss
    end

    update!(Q, gs)
end
