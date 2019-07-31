export ExperienceBasedSampleModel, sample, update!

import StatsBase:sample

"""
      ExperienceBasedSampleModel <: AbstractSampleBasedModel
Generate a turn sample based on previous experiences.
"""
struct ExperienceBasedSampleModel <: AbstractSampleBasedModel
   experiences::Dict{Any, Dict{Any, NamedTuple{(:reward, :terminal, :nextstate), Tuple{Float64, Bool, Any}}}}
   ExperienceBasedSampleModel() = new(Dict{Any, Dict{Any, NamedTuple{(:reward, :terminal, :nextstate), Tuple{Float64, Bool, Any}}}}())
end

function update!(m::ExperienceBasedSampleModel, s, a, r, d, s′)
   if haskey(m.experiences, s)
         m.experiences[s][a] = (reward=r, terminal=d, nextstate=s′)
   else
         m.experiences[s] = Dict{Any, NamedTuple{(:reward, :terminal, :nextstate), Tuple{Float64, Bool, Any}}}(a => (reward=r, terminal=d, nextstate=s′))
   end
end

function sample(model::ExperienceBasedSampleModel)
    s = rand(keys(model.experiences))
    a = rand(keys(model.experiences[s]))
    s, a, model.experiences[s][a]...
end
