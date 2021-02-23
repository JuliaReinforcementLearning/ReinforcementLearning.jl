export TimeBasedSamplingModel

using Random
import StatsBase: sample

"""
    TimeBasedSamplingModel(n_actions::Int, κ::Float64 = 1e-4)
"""
Base.@kwdef mutable struct TimeBasedSamplingModel{R} <: AbstractEnvironmentModel
    experiences::Dict{Any,Dict{Any,Any}} = Dict{Any,Dict{Any,Any}}()
    n_actions::Int
    κ::Float64 = 1e-4
    t::Int = 0
    last_visit::Dict{Tuple{Any,Any},Int} = Dict{Tuple{Any,Any},Int}()
    rng::R = Random.GLOBAL_RNG
end

function RLBase.update!(
    m::TimeBasedSamplingModel,
    t::AbstractTrajectory,
    ::AbstractPolicy,
    ::AbstractEnv,
    ::Union{PreActStage,PostEpisodeStage},
)
    if length(t[:terminal]) > 0
        transition = (
            t[:state][end-1],
            t[:action][end-1],
            t[:reward][end],
            t[:terminal][end],
            t[:state][end],
        )
        update!(m, transition)
    end
end

function RLBase.update!(m::TimeBasedSamplingModel, transition::Tuple)
    s, a, r, d, s′ = transition
    if haskey(m.experiences, s)
        m.experiences[s][a] = (reward = r, terminal = d, nextstate = s′)
    else
        m.experiences[s] = Dict(a => (reward = r, terminal = d, nextstate = s′))
    end
    m.t += 1
    m.last_visit[(s, a)] = m.t
end

sample(model::TimeBasedSamplingModel) = sample(model.rng, model)

function sample(rng::AbstractRNG, m::TimeBasedSamplingModel)
    if length(m.experiences) > 0
        s = rand(rng, keys(m.experiences))
        a = rand(rng, 1:m.n_actions)
        r, d, s′ = get(m.experiences[s], a, (0.0, false, s))
        r += m.κ * sqrt(m.t - get(m.last_visit, (s, a), 0))
        s, a, r, d, s′
    else
        nothing
    end
end
