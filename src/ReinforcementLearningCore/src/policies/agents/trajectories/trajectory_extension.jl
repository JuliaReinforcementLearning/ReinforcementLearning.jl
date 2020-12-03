export NStepInserter, BatchSampler, NStepBatchSampler

using Random

#####
# Inserters
#####

abstract type AbstractInserter end

Base.@kwdef struct NStepInserter <: AbstractInserter
    n::Int = 1
end

function Base.push!(
    t::CircularVectorSARTSATrajectory,
    𝕥::CircularArraySARTTrajectory,
    inserter::NStepInserter,
)
    N = length(𝕥)
    n = inserter.n
    for i in 1:(N-n+1)
        for k in SART
            push!(t[k], select_last_dim(𝕥[k], i))
        end
        push!(t[:next_state], select_last_dim(𝕥[:state], i+n))
        push!(t[:next_action], select_last_dim(𝕥[:action], i+n))
    end
end

#####
# Samplers
#####

abstract type AbstractSampler{traces} end

"""
    sample([rng=Random.GLOBAL_RNG], trajectory, sampler, [traces=Val(keys(trajectory))])

!!! note
    Here we return a copy instead of a view:
    1. Each sample is independent of the original `trajectory` so that `trajectory` can be updated async.
    2. [Copy is not always so bad](https://docs.julialang.org/en/v1/manual/performance-tips/#Copying-data-is-not-always-bad).
"""
function StatsBase.sample(t::AbstractTrajectory, sampler::AbstractSampler)
    sample(Random.GLOBAL_RNG, t, sampler)
end

#####
## BatchSampler
#####

struct BatchSampler{traces} <: AbstractSampler{traces}
    batch_size::Int
end

BatchSampler(batch_size::Int) = BatchSampler{SARTSA}(batch_size)

function StatsBase.sample(rng::AbstractRNG, t::AbstractTrajectory, s::BatchSampler)
    inds = rand(rng, 1:length(t), s.batch_size)
    inds, select(inds, t, s)
end

function select(inds::Vector{Int}, t::CircularVectorSARTSATrajectory, s::BatchSampler{traces}) where traces
    NamedTuple{SARTSA}(Flux.batch(view(t[x], inds)) for x in traces)
end

function select(inds::Vector{Int}, t::CircularArraySARTTrajectory, s::BatchSampler{SARTS})
    NamedTuple{SARTS}((
        (convert(Array, consecutive_view(t[x], inds)) for x in SART)...,
        convert(Array,consecutive_view(t[:state], inds.+1))
    ))
end

#####
## NStepBatchSampler
#####

Base.@kwdef struct NStepBatchSampler{traces} <: AbstractSampler{traces}
    γ::Float32
    n::Int = 1
    batch_size::Int = 32
    stack_size::Union{Nothing,Int} = nothing
end

function StatsBase.sample(rng::AbstractRNG, t::AbstractTrajectory, s::NStepBatchSampler)
    inds = rand(rng, 1:(length(t)-s.n+1), s.batch_size)
    inds, select(inds, t, s)
end

function StatsBase.sample(rng::AbstractRNG, t::PrioritizedTrajectory{<:SumTree}, s::NStepBatchSampler)
    bz, sz = s.batch_size, s.stack_size
    inds = Vector{Int}(undef, bz)
    priorities = Vector{Float32}(undef, bz)
    valid_ind_range = isnothing(sz) ? (1:(length(t)-s.n+1)) : (sz:(length(t)-s.n+1))
    for i in 1:bz
        ind, p = sample(rng, t.priority)
        while ind ∉ valid_ind_range
            ind, p = sample(rng, t.priority)
        end
        inds[i] = ind
        priorities[i] = p
    end
    inds, (priority=priorities, select(inds, t.traj, s)...)
end

function select(inds::Vector{Int}, traj::CircularArraySARTTrajectory, s::NStepBatchSampler{traces}) where traces
    γ, n, bz, sz = s.γ, s.n, s.batch_size, s.stack_size
    next_inds = inds .+ n

    s = convert(Array, consecutive_view(traj[:state], inds;n_stack = sz))
    a = convert(Array, consecutive_view(traj[:action], inds))
    s′ = convert(Array, consecutive_view(traj[:state], next_inds;n_stack = sz))

    consecutive_rewards = consecutive_view(traj[:reward], inds; n_horizon = n)
    consecutive_terminals = consecutive_view(traj[:terminal], inds; n_horizon = n)
    r, t = zeros(Float32, bz), fill(false, bz)

    # make sure that we only consider experiences in current episode
    for i in 1:bz
        m = findfirst(view(consecutive_terminals, :, i))
        if isnothing(m)
            t[i] = false
            r[i] = discount_rewards_reduced(view(consecutive_rewards, :, i), γ)
        else
            t[i] = true
            r[i] = discount_rewards_reduced(view(consecutive_rewards, 1:m, i), γ)
        end
    end

    if traces == SARTS
        NamedTuple{SARTS}((s, a, r, t, s′))
    elseif traces == SLARTSL
        l = convert(Array, consecutive_view(traj[:legal_actions_mask], inds))
        l′ = convert(Array, consecutive_view(traj[:next_legal_actions_mask], next_inds))
        NamedTuple{SLARTSL}((s, l, a, r, t, s′, l′))
    else
        @error "unsupported traces $traces"
    end
end