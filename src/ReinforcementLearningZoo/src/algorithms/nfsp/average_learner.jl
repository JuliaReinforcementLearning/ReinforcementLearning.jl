export AverageLearner

mutable struct AverageLearner{
    Tq<:AbstractApproximator,
    R<:AbstractRNG,
} <: AbstractLearner
    approximator::Tq
    min_reservoir_history::Int
    update_freq::Int
    update_step::Int
    sampler::NStepBatchSampler
    rng::R
end

"""
    AverageLearner(;kwargs...)

In the `Neural Fictitious Self-play` algorithm, AverageLearner, also known as  Supervisor Learner, works to learn the best response for the state from RL_agent's policy.

See paper: [Deep Reinforcement Learning from Self-Play in Imperfect-Information Games](https://arxiv.org/pdf/1603.01121.pdf)

# Keywords

- `approximator`::[`AbstractApproximator`](@ref).
- `batch_size::Int=32`
- `update_horizon::Int=1`: length of update ('n' in n-step update).
- `min_reservoir_history::Int=32`: number of transitions that should be experienced before updating the `approximator`.
- `update_freq::Int=1`: the frequency of updating the `approximator`.
- `stack_size::Union{Int, Nothing}=nothing`: use the recent `stack_size` frames to form a stacked state.
- `traces = SARTS`.
- `rng = Random.GLOBAL_RNG`
"""

function AverageLearner(;
    approximator::Tq,
    batch_size::Int = 32,
    update_horizon::Int = 1,
    min_reservoir_history::Int = 32,
    update_freq::Int = 1,
    update_step::Int = 0,
    stack_size::Union{Int,Nothing} = nothing,
    traces = SARTS,
    rng = Random.GLOBAL_RNG,
) where {Tq}
    sampler = NStepBatchSampler{traces}(;
        γ = 0f0, # no need to set discount factor
        n = update_horizon,
        stack_size = stack_size,
        batch_size = batch_size,
    )
    AverageLearner(
        approximator,
        min_reservoir_history,
        update_freq,
        update_step,
        sampler,
        rng,
    )
end

Flux.functor(x::AverageLearner) = (Q = x.approximator, ), y -> begin
    x = @set x.approximator = y.Q
    x
end

function (learner::AverageLearner)(env)
    env |>
    state |>
    x -> Flux.unsqueeze(x, ndims(x) + 1) |>
    x -> send_to_device(device(learner), x) |>
    learner.approximator |>
    send_to_host |> vec
end

function RLBase.update!(learner::AverageLearner, t::AbstractTrajectory)
    length(t[:terminal]) - learner.sampler.n <= learner.min_reservoir_history && return

    learner.update_step += 1
    learner.update_step % learner.update_freq == 0 || return

    inds, batch = sample(learner.rng, t, learner.sampler)
    if t isa PrioritizedTrajectory
        priorities = update!(learner, batch)
        t[:priority][inds] .= priorities
    else
        update!(learner, batch)
    end
end

function RLBase.update!(learner::AverageLearner, batch::NamedTuple)
    Q = learner.approximator
    _device(x) = send_to_device(device(Q), x)

    local s, a
    @sync begin
        @async s = _device(batch[:state])
        @async a = _device(batch[:action])
    end

    gs = gradient(params(Q)) do
        ŷ = Q(s)
        y = Flux.onehotbatch(a, axes(ŷ, 1)) |> _device
        crossentropy(ŷ, y)
    end

    update!(Q, gs)  
end