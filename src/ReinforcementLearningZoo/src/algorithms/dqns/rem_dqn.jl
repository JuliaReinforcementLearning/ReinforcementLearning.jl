export REMDQNLearner

mutable struct REMDQNLearner{
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
    ensemble_num::Int
    ensemble_method::Symbol
    rng::R
    # for logging
    loss::Float32
end

"""
    REMDQNLearner(;kwargs...)

See paper: [An Optimistic Perspective on Offline Reinforcement Learning](https://arxiv.org/abs/1907.04543)

# Keywords

- `approximator`::[`AbstractApproximator`](@ref): used to get Q-values of a state.
- `target_approximator`::[`AbstractApproximator`](@ref): similar to `approximator`, but used to estimate the target (the next state).
- `loss_func`: the loss function.
- `γ::Float32=0.99f0`: discount rate.
- `batch_size::Int=32`
- `update_horizon::Int=1`: length of update ('n' in n-step update).
- `min_replay_history::Int=32`: number of transitions that should be experienced before updating the `approximator`.
- `update_freq::Int=4`: the frequency of updating the `approximator`.
- `ensemble_num::Int=1`: the number of ensemble approximators.
- `ensemble_method::Symbol=:rand`: the method of combining Q values. ':rand' represents random ensemble mixture, and ':mean' is the average.
- `target_update_freq::Int=100`: the frequency of syncing `target_approximator`.
- `stack_size::Union{Int, Nothing}=4`: use the recent `stack_size` frames to form a stacked state.
- `traces = SARTS`, set to `SLARTSL` if you are to apply to an environment of `FULL_ACTION_SET`.
- `rng = Random.GLOBAL_RNG`
"""
function REMDQNLearner(;
    approximator::Tq,
    target_approximator::Tt,
    loss_func::Tf,
    stack_size::Union{Int,Nothing} = nothing,
    γ::Float32 = 0.99f0,
    batch_size::Int = 32,
    update_horizon::Int = 1,
    min_replay_history::Int = 32,
    update_freq::Int = 1,
    ensemble_num::Int = 1,
    ensemble_method::Symbol = :rand,
    target_update_freq::Int = 100,
    traces = SARTS,
    update_step = 0,
    rng = Random.GLOBAL_RNG,
) where {Tq,Tt,Tf}
    copyto!(approximator, target_approximator)
    sampler = NStepBatchSampler{traces}(;
        γ = γ,
        n = update_horizon,
        stack_size = stack_size,
        batch_size = batch_size,
    )
    REMDQNLearner(
        approximator,
        target_approximator,
        loss_func,
        min_replay_history,
        update_freq,
        update_step,
        target_update_freq,
        sampler,
        ensemble_num,
        ensemble_method,
        rng,
        0.0f0,
    )
end

Flux.functor(x::REMDQNLearner) = (Q = x.approximator, Qₜ = x.target_approximator),
y -> begin
    x = @set x.approximator = y.Q
    x = @set x.target_approximator = y.Qₜ
    x
end

function (learner::REMDQNLearner)(env)
    s = send_to_device(device(learner.approximator), state(env))
    s = Flux.unsqueeze(s, ndims(s) + 1)
    q = reshape(learner.approximator(s), :, learner.ensemble_num)
    vec(mean(q, dims = 2)) |> send_to_host
end

function RLBase.update!(learner::REMDQNLearner, batch::NamedTuple)
    Q = learner.approximator
    Qₜ = learner.target_approximator
    γ = learner.sampler.γ
    loss_func = learner.loss_func
    n = learner.sampler.n
    batch_size = learner.sampler.batch_size
    ensemble_num = learner.ensemble_num
    D = device(Q)
    # Build a convex polygon to make a combination of multiple Q-value estimates as a Q-value estimate.
    if learner.ensemble_method == :rand
        convex_polygon = rand(Float32, (1, ensemble_num))
    else
        convex_polygon = ones(Float32, (1, ensemble_num))
    end
    convex_polygon ./= sum(convex_polygon)
    convex_polygon = send_to_device(D, convex_polygon)

    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    a = CartesianIndex.(a, 1:batch_size)

    target_q = Qₜ(s′)
    target_q = convex_polygon .* reshape(target_q, :, ensemble_num, batch_size)
    target_q = dropdims(sum(target_q, dims=2), dims=2)

    if haskey(batch, :next_legal_actions_mask)
        l′ = send_to_device(D, batch[:next_legal_actions_mask])
        target_q .+= ifelse.(l′, 0.0f0, typemin(Float32))
    end

    q′ = dropdims(maximum(target_q; dims = 1), dims = 1)
    G = r .+ γ^n .* (1 .- t) .* q′

    gs = gradient(params(Q)) do
        q = Q(s)
        q = convex_polygon .* reshape(q, :, ensemble_num, batch_size)
        q = dropdims(sum(q, dims=2), dims=2)[a]

        loss = loss_func(G, q)
        ignore() do
            learner.loss = loss
        end
        loss
    end

    update!(Q, gs)
end 

