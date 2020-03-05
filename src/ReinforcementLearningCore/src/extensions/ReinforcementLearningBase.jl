using CuArrays
using Distributions: pdf
using Random

RLBase.get_prob(p::AbstractPolicy, obs, ::RLBase.AbstractActionStyle, a) = pdf(get_prob(p, obs), a)

Random.rand(s::MultiContinuousSpace{<:CuArray}) = rand(CuArrays.CURAND.generator(), s)