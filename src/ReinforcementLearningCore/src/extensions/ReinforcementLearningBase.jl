export @policy_str

using CuArrays
using Distributions: pdf
using Random
using Flux
using BSON

RLBase.get_prob(p::AbstractPolicy, obs, ::RLBase.AbstractActionStyle, a) =
    pdf(get_prob(p, obs), a)

Random.rand(s::MultiContinuousSpace{<:CuArray}) = rand(CuArrays.CURAND.generator(), s)

# avoid fallback silently
Flux.testmode!(p::AbstractPolicy, mode = true) =
    @error "someone forgets to implement this method!!!"

macro policy_str(path)
    load_policy(path)
end

function load_policy(path)
    if isdir(path)
        path = joinpath(path, "policy.bson")
    end
    BSON.load(path)[:policy]
end
