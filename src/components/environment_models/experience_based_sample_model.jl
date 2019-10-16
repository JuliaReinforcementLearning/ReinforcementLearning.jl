export ExperienceBasedSampleModel, sample, update!

import StatsBase: sample

"""
    ExperienceBasedSampleModel() -> ExperienceBasedSampleModel

Generate a transition based on previous experiences.
"""
mutable struct ExperienceBasedSampleModel <: AbstractSampleBasedModel
    experiences::Dict{
        Any,
        Dict{Any,NamedTuple{(:reward, :terminal, :nextstate),Tuple{Float64,Bool,Any}}},
    }
    sample_count::Int
    ExperienceBasedSampleModel() =
        new(
            Dict{
                Any,
                Dict{
                    Any,
                    NamedTuple{(:reward, :terminal, :nextstate),Tuple{Float64,Bool,Any}},
                },
            }(),
            0,
        )
end

function extract_transitions(buffer::EpisodeTurnBuffer, m::ExperienceBasedSampleModel)
    if length(buffer) > 0
        state(buffer)[end-1],
        action(buffer)[end-1],
        reward(buffer)[end],
        terminal(buffer)[end],
        state(buffer)[end]
    else
        nothing
    end
end

function update!(m::ExperienceBasedSampleModel, transition::Tuple)
    s, a, r, d, s′ = transition
    if haskey(m.experiences, s)
        m.experiences[s][a] = (reward = r, terminal = d, nextstate = s′)
    else
        m.experiences[s] = Dict{
            Any,
            NamedTuple{(:reward, :terminal, :nextstate),Tuple{Float64,Bool,Any}},
        }(a => (reward = r, terminal = d, nextstate = s′))
    end
end

function sample(model::ExperienceBasedSampleModel)
    s = rand(keys(model.experiences))
    a = rand(keys(model.experiences[s]))
    model.sample_count += 1
    s, a, model.experiences[s][a]...
end
