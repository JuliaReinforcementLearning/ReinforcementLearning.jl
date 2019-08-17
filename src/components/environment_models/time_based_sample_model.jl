export TimeBasedSampleModel, update!, sample

import StatsBase:sample

mutable struct TimeBasedSampleModel <: AbstractSampleBasedModel
    experiences::Dict{Any, Dict{Any, NamedTuple{(:reward, :terminal, :nextstate), Tuple{Float64, Bool, Any}}}}
    nactions::Int
    κ::Float64
    t::Int
    last_visit::Dict{Tuple{Any, Any}, Int}
    TimeBasedSampleModel(nactions::Int, κ::Float64=1e-4) = new(Dict{Any, Dict{Any, NamedTuple{(:reward, :terminal, :nextstate), Tuple{Float64, Bool, Any}}}}(), nactions, κ, 0, Dict{Tuple{Any, Any}, Int}())
end

function update!(m::TimeBasedSampleModel, s, a, r, d, s′)
   if haskey(m.experiences, s)
         m.experiences[s][a] = (reward=r, terminal=d, nextstate=s′)
   else
         m.experiences[s] = Dict{Any, NamedTuple{(:reward, :terminal, :nextstate), Tuple{Float64, Bool, Any}}}(a => (reward=r, terminal=d, nextstate=s′))
   end
   m.t += 1
   m.last_visit[(s,a)] = m.t
end

function sample(m::TimeBasedSampleModel)
   s = rand(keys(m.experiences))
   a = rand(1:m.nactions)
   r, d, s′ = get(m.experiences[s], a, (0., false, s))
   r += m.κ * sqrt(m.t - get(m.last_visit, (s, a), 0))
   s, a, r, d, s′
end
