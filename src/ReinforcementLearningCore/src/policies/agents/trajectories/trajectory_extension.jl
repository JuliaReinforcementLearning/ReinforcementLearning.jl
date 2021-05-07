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
        push!(t[:next_state], select_last_dim(𝕥[:state], i + n))
        push!(t[:next_action], select_last_dim(𝕥[:action], i + n))
    end
end

#####
# Samplers
#####

abstract type AbstractSampler{traces} end

# TODO: deprecate this method with `(s::AbstractSampler)(traj)` instead

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

# TODO: add an async batch sampler to pre-fetch next batch

#####
## BatchSampler
#####

mutable struct BatchSampler{traces} <: AbstractSampler{traces}
    batch_size::Int
    cache::Any
    rng::Any
end

BatchSampler(batch_size::Int; cache = nothing, rng = Random.GLOBAL_RNG) =
    BatchSampler{SARTSA}(batch_size, cache, rng)
BatchSampler{T}(batch_size::Int; cache = nothing, rng = Random.GLOBAL_RNG) where {T} =
    BatchSampler{T}(batch_size, cache, rng)

(s::BatchSampler)(t::AbstractTrajectory) = sample(s.rng, t, s)

# TODO: deprecate
function StatsBase.sample(rng::AbstractRNG, t::AbstractTrajectory, s::BatchSampler)
    inds = rand(rng, 1:length(t), s.batch_size)
    fetch!(s, t, inds)
    inds, s.cache
end

function fetch!(s::BatchSampler, t::AbstractTrajectory, inds::Vector{Int})
    batch = NamedTuple{keys(t)}(view(t[x], inds) for x in keys(t))
    if isnothing(s.cache)
        s.cache = map(Flux.batch, batch)
    else
        map(s.cache, batch) do dest, src
            batch!(dest, src)
        end
    end
end

function fetch!(s::BatchSampler{SARTS}, t::CircularArraySARTTrajectory, inds::Vector{Int})
    batch = NamedTuple{SARTS}((
        (consecutive_view(t[x], inds) for x in SART)...,
        consecutive_view(t[:state], inds .+ 1),
    ))
    if isnothing(s.cache)
        s.cache = map(batch) do x
            convert(Array, x)
        end
    else
        map(s.cache, batch) do dest, src
            copyto!(dest, src)
        end
    end
end

#####
## NStepBatchSampler
#####

Base.@kwdef mutable struct NStepBatchSampler{traces} <: AbstractSampler{traces}
    γ::Float32
    n::Int = 1
    batch_size::Int = 32
    stack_size::Union{Nothing,Int} = nothing
    rng::Any = Random.GLOBAL_RNG
    cache::Any = nothing
end

# TODO:deprecate
function StatsBase.sample(rng::AbstractRNG, t::AbstractTrajectory, s::NStepBatchSampler)
    valid_range =
        isnothing(s.stack_size) ? (1:(length(t)-s.n+1)) : (s.stack_size:(length(t)-s.n+1))
    inds = rand(rng, valid_range, s.batch_size)
    inds, fetch!(s, t, inds)
end

function StatsBase.sample(rng::AbstractRNG, t::PrioritizedTrajectory, s::NStepBatchSampler)
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
    inds, (priority = priorities, fetch!(s, t.traj, inds)...)
end

function fetch!(
    sampler::NStepBatchSampler{traces},
    traj::Union{CircularArraySARTTrajectory,CircularArraySLARTTrajectory},
    inds::Vector{Int},
) where {traces}
    γ, n, bz, sz = sampler.γ, sampler.n, sampler.batch_size, sampler.stack_size
    cache = sampler.cache
    next_inds = inds .+ n

    s = consecutive_view(traj[:state], inds; n_stack = sz)
    a = consecutive_view(traj[:action], inds)
    s′ = consecutive_view(traj[:state], next_inds; n_stack = sz)

    consecutive_rewards = consecutive_view(traj[:reward], inds; n_horizon = n)
    consecutive_terminals = consecutive_view(traj[:terminal], inds; n_horizon = n)
    r = isnothing(cache) ? zeros(Float32, bz) : cache.reward
    t = isnothing(cache) ? fill(false, bz) : cache.terminal

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
        batch = NamedTuple{SARTS}((s, a, r, t, s′))
    elseif traces == SLARTSL
        l = consecutive_view(traj[:legal_actions_mask], inds)
        l′ = consecutive_view(traj[:next_legal_actions_mask], next_inds)
        batch = NamedTuple{SLARTSL}((s, l, a, r, t, s′, l′))
    else
        @error "unsupported traces $traces"
    end

    if isnothing(sampler.cache)
        sampler.cache = map(batch) do x
            convert(Array, x)
        end
    else
        map(sampler.cache, batch) do dest, src
            copyto!(dest, src)
        end
    end
end
