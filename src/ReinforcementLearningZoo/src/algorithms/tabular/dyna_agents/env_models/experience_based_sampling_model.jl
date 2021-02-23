export ExperienceBasedSamplingModel, sample

using Random
import StatsBase: sample

"""
    ExperienceBasedSamplingModel

Randomly generate a transition of (s, a, r, t, s′) based on previous experiences
in each sampling.
"""
Base.@kwdef mutable struct ExperienceBasedSamplingModel{R} <: AbstractEnvironmentModel
    experiences::Dict{Any,Dict{Any,Any}} = Dict{Any,Dict{Any,Any}}()
    sample_count::Int = 0
    rng::R = Random.GLOBAL_RNG
end

function RLBase.update!(
    m::ExperienceBasedSamplingModel,
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

function RLBase.update!(m::ExperienceBasedSamplingModel, transition::Tuple)
    s, a, r, d, s′ = transition
    if haskey(m.experiences, s)
        m.experiences[s][a] = (reward = r, terminal = d, nextstate = s′)
    else
        m.experiences[s] = Dict(a => (reward = r, terminal = d, nextstate = s′))
    end
end

sample(model::ExperienceBasedSamplingModel) = sample(model.rng, model)

function sample(rng::AbstractRNG, model::ExperienceBasedSamplingModel)
    if length(model.experiences) > 0
        s = rand(rng, keys(model.experiences))
        a = rand(rng, keys(model.experiences[s]))
        model.sample_count += 1
        s, a, model.experiences[s][a]...
    else
        nothing
    end
end
