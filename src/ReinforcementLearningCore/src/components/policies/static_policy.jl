export StaticPolicy

using MacroTools: @forward

"""
    StaticPolicy(policy)

Create a policy wrapper so that it will do nothing when calling
`update!(policy::StaticPolicy, args...)`. Usually used in the
distributed mode as a worker.
"""
struct StaticPolicy{P<:AbstractPolicy} <: AbstractPolicy
    p::P
end

(π::StaticPolicy)(env) = π.p(env)

@forward StaticPolicy.p RLBase.get_priority, RLBase.get_prob

RLBase.update!(p::StaticPolicy, args...) = nothing

RLBase.update!(p::StaticPolicy, ps::Params) = update!(p.p, ps)

Flux.@functor StaticPolicy
