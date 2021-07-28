export PrioritizedDQNLearner

"""
    PrioritizedDQNLearner(;kwargs...)

See paper: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
And also https://danieltakeshi.github.io/2019/07/14/per/

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
- `default_priority::Float64=100.`: the default priority for newly added transitions.
- `rng = Random.GLOBAL_RNG`

!!! note
    Our implementation is slightly different from the original paper. But it should be aligned with the version in [dopamine](https://github.com/google/dopamine/blob/90527f4eaad4c574b92df556c02dea45853ffd2e/dopamine/jax/agents/rainbow/rainbow_agent.py#L26-L30).
"""
mutable struct PrioritizedDQNLearner{
    Tq<:AbstractApproximator,
    Tt<:AbstractApproximator,
    Tf,
    R<:AbstractRNG,
} <: AbstractLearner
    approximator::Tq
    target_approximator::Tt
    loss_func::Tf
    sampler::NStepBatchSampler
    min_replay_history::Int
    update_freq::Int
    target_update_freq::Int
    update_step::Int
    default_priority::Float32
    β_priority::Float32
    rng::R
    # for logging
    loss::Float32
end

function PrioritizedDQNLearner(;
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
    update_step::Int = 0,
    default_priority::Float32 = 100.0f0,
    β_priority::Float32 = 0.5f0,
    traces = SARTS,
    rng = Random.GLOBAL_RNG,
) where {Tq,Tt,Tf}
    copyto!(approximator, target_approximator)
    sampler = NStepBatchSampler{traces}(;
        γ = γ,
        n = update_horizon,
        stack_size = stack_size,
        batch_size = batch_size,
    )
    PrioritizedDQNLearner(
        approximator,
        target_approximator,
        loss_func,
        sampler,
        min_replay_history,
        update_freq,
        target_update_freq,
        update_step,
        default_priority,
        β_priority,
        rng,
        0.0f0,
    )
end


Flux.functor(x::PrioritizedDQNLearner) = (Q = x.approximator, Qₜ = x.target_approximator),
y -> begin
    x = @set x.approximator = y.Q
    x = @set x.target_approximator = y.Qₜ
    x
end

"""

!!! note
    The state of the observation is assumed to have been stacked,
    if `!isnothing(stack_size)`.
"""
function (learner::PrioritizedDQNLearner)(env)
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

function RLBase.update!(learner::PrioritizedDQNLearner, batch::NamedTuple)
    Q = learner.approximator
    Qₜ = learner.target_approximator
    γ = learner.sampler.γ
    β = learner.β_priority
    loss_func = learner.loss_func
    n = learner.sampler.n
    batch_size = learner.sampler.batch_size

    D = device(Q)
    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    a = CartesianIndex.(a, 1:batch_size)

    updated_priorities = Vector{Float32}(undef, batch_size)
    w = 1.0f0 ./ ((batch.priority .+ 1f-10) .^ β)
    w ./= maximum(w)
    w = send_to_device(D, w)

    target_q = Qₜ(s′)
    if haskey(batch, :next_legal_actions_mask)
        l′ = send_to_device(D, batch[:next_legal_actions_mask])
        target_q .+= ifelse.(l′, 0.0f0, typemin(Float32))
    end

    q′ = dropdims(maximum(target_q; dims = 1), dims = 1)
    G = r .+ γ^n .* (1 .- t) .* q′

    gs = gradient(params(Q)) do
        q = Q(s)[a]
        batch_losses = loss_func(G, q)
        loss = dot(vec(w), vec(batch_losses)) * 1 // batch_size
        ignore() do
            updated_priorities .= send_to_host(vec((batch_losses .+ 1f-10) .^ β))
            learner.loss = loss
        end
        loss
    end

    update!(Q, gs)
    updated_priorities
end
