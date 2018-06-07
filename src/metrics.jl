"""
    mutable struct MeanReward 
        meanreward::Float64
        counter::Int64

Computes iteratively the mean reward.
"""
mutable struct MeanReward
    meanreward::Float64
    counter::Int64
end
"""
    MeanReward()

Initializes `counter` and `meanreward` to 0.
"""
MeanReward() = MeanReward(0., 0)
function callback!(p::MeanReward, rlsetup, sraw, a, r, done)
    p.counter += 1
    p.meanreward += 1/p.counter * (r - p.meanreward)
end
function reset!(p::MeanReward)
    p.counter = 0
    p.meanreward = 0.
end
getvalue(p::MeanReward) = p.meanreward
export MeanReward, getvalue

"""
    mutable struct TotalReward 
        reward::Float64

Accumulates all rewards.
"""
mutable struct TotalReward
    reward::Float64
end
"""
    TotalReward()

Initializes `reward` to 0.
"""
TotalReward() = TotalReward(0.)
function callback!(p::TotalReward, rlsetup, sraw, a, r, done)
    p.reward += r
end
function reset!(p::TotalReward)
    p.reward = 0.
end
getvalue(p::TotalReward) = p.reward
export TotalReward

"""
    mutable struct TimeSteps
        counter::Int64

Counts the number of timesteps the simulation is running.
"""
mutable struct TimeSteps
    counter::Int64
end
"""
    TimeSteps()

Initializes `counter` to 0.
"""
TimeSteps() = TimeSteps(0)
function callback!(p::TimeSteps, rlsetup, sraw, a, r, done)
    p.counter += 1
end
function reset!(p::TimeSteps)
    p.counter = 0
end
getvalue(p::TimeSteps) = p.counter
export TimeSteps

"""
    EvaluationPerEpisode
        values::Array{Float64, 1}
        metric::SimpleEvaluationMetric

Stores the value of the simple `metric` for each episode in `values`.
"""
struct EvaluationPerEpisode{T}
    values::Array{Float64, 1}
    metric::T
    returnmean::Bool
end
"""
    EvaluationPerEpisode(metric = MeanReward())

Initializes with empty `values` array and simple `metric` (default
[`MeanReward`](@ref)).
Other options are [`TimeSteps`](@ref) (to measure the lengths of episodes) or
[`TotalReward`](@ref).
"""
function EvaluationPerEpisode(metric = MeanReward(); returnmean = false)
    EvaluationPerEpisode(Float64[], metric,returnmean)
end
function callback!(p::EvaluationPerEpisode, rlsetup, sraw, a, r, done)
    callback!(p.metric, rlsetup, sraw, a, r, done)
    if done
        push!(p.values, getvalue(p.metric))
        reset!(p.metric)
    end
end
function reset!(p::EvaluationPerEpisode)
    reset!(p.metric)
    empty!(p.values)
end
export EvaluationPerEpisode

"""
    EvaluationPerT
        T::Int64
        counter::Int64
        values::Array{Float64, 1}
        metric::SimpleEvaluationMetric

Stores the value of the simple `metric` after every `T` steps in `values`.
"""
mutable struct EvaluationPerT{T}
    T::Int64
    counter::Int64
    values::Array{Float64, 1}
    metric::T
    returnmean::Bool
end
"""
    EvaluationPerT(T, metric = MeanReward())

Initializes with `T`, `counter` = 0, empty `values` array and simple `metric`
(default [`MeanReward`](@ref)).  Another option is [`TotalReward`](@ref).
"""
function EvaluationPerT(T, metric = MeanReward(); returnmean = false)
    EvaluationPerT(T, 0, Float64[], metric, returnmean)
end
function callback!(p::EvaluationPerT, rlsetup, sraw, a, r, done)
    callback!(p.metric, rlsetup, sraw, a, r, done)
    p.counter += 1
    if p.counter == p.T
        push!(p.values, getvalue(p.metric))
        reset!(p.metric)
        p.counter = 0
    end
end
function reset!(p::EvaluationPerT)
    reset!(p.metric)
    p.counter = 0
    empty!(p.values)
end
getvalue(p::Union{EvaluationPerEpisode, EvaluationPerT}) = 
    p.returnmean ? mean(p.values) : p.values
export EvaluationPerT
function getlastvaluestring(p::Union{EvaluationPerT, EvaluationPerEpisode})
    if length(p.values) > 0
        "$(typeof(p.metric).name.name): $(p.values[end])"
    else
        ""
    end
end

