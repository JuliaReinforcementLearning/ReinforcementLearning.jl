export SoftmaxPolicy, EpsilonGreedyPolicy, ForcedPolicy,
       getactionprobabilities

"""
    mutable struct SoftmaxPolicy <: AbstractSoftmaxPolicy
        β::Float64

Choose action ``a`` with probability

```math
\\frac{e^{\\beta x_a}}{\\sum_{a'} e^{\\beta x_{a'}}}
```

where ``x`` is a vector of values for each action. In states with actions that
were never chosen before, a uniform random novel action is returned.

    SoftmaxPolicy(; β = 1.)

Returns a SoftmaxPolicy with default β = 1.
"""
mutable struct SoftmaxPolicy{T} 
    β::Float64
    π::T
end
SoftmaxPolicy(π; β = 1.) = SoftmaxPolicy(float(β), π)
function (p::SoftmaxPolicy)(s)
    values = p.π(s)
    if maximum(values) == Inf64
        rand(findall(v -> v == Inf64, values))
    else
        samplesoftmaxaction(p, values)
    end
end

# Samples from Categorical(exp(input)/sum(exp(input)))
function samplesoftmaxaction(policy::SoftmaxPolicy, values)
    if policy.β == Inf
        samplegreedyaction(policy, values)
    else
        StatsBase.wsample(exp.(policy.β .* values))
    end
end

function getactionprobabilities(policy::SoftmaxPolicy, state)
    values = policy.π(state)
    if maximum(values) == Inf || policy.β == Inf
        p = zeros(length(values))
        vmax = maximum(values)
        a = findall(v -> v == vmax, values)
        for i in a
            p[i] = 1/length(a)
        end
        return p
    else
        expvals = exp.(policy.β .* (values .- maximum(values)))
        return expvals/sum(expvals)
    end
end


"""
    mutable struct EpsilonGreedyPolicy{kind}
        ϵ::Float64

Chooses the action with the highest value with probability `1 - ϵ` and selects 
an action uniformly random with probability `ϵ`.
"""
mutable struct EpsilonGreedyPolicy{kind, Ta, Tf}
    ϵ::Float64
    actionspace::Ta
    Q::Tf
end
function EpsilonGreedyPolicy(ϵ, actionspace::Ta, Q::Tf; 
                             kind = :veryoptimistic) where {Ta, Tf}
    EpsilonGreedyPolicy{kind, Ta, Tf}(ϵ, actionspace, Q)
end
(p::EpsilonGreedyPolicy)(s) = rand() < p.ϵ ? sample(p.actionspace) : 
                                             samplegreedyaction(p, p.Q(s))


import Base.maximum, Base.isequal
maximum(::EpsilonGreedyPolicy, v) = maximumbelowInf(v)
maximum(::SoftmaxPolicy, v) = maximum(v)
isequal(::EpsilonGreedyPolicy{:optimistic, Ta, Tf}, v1, v2) where {Ta, Tf} = v1 >= v2
maximum(::EpsilonGreedyPolicy{:veryoptimistic, Ta, Tf}, v) where {Ta, Tf} = maximum(v)
isequal(::EpsilonGreedyPolicy, v1, v2) = v1 == v2
isequal(::SoftmaxPolicy, v1, v2) = v1 == v2

samplegreedyaction(p, a::Int) = a # needed by mdplearner
function samplegreedyaction(policy, values)
    vmax = maximum(policy, values)
    c = 1
    a = 1
    for (i, v) in enumerate(values)
        if isequal(policy, v, vmax)
            if rand() < 1/c
                a = i
            end
            c += 1
        end
    end
    a
end

function getactionprobabilities(policy::EpsilonGreedyPolicy, state)
    values = policy.Q(state)
    p = ones(length(values))/length(values) * policy.ϵ
    vmax = maximum(policy, values)
    c = 0
    for v in values
        if isequal(policy, v, vmax)
            c += 1
        end
    end
    p2 = (1. - policy.ϵ)/c
    for (i, v) in enumerate(values)
        if isequal(policy, v, vmax)
            p[i] += p2
        end
    end
    p
end

struct NMarkovPolicy{N, Tpol, Tbuf}
    policy::Tpol
    buffer::Tbuf
end
function NMarkovPolicy(N, pol::Tpol, buf::Tbuf) where {Tpol, Tbuf}
    NMarkovPolicy{N, Tpol, Tbuf}(pol, buf)
end
function (p::NMarkovPolicy{N, Tpol, Tbuf})(s) where {N, Tpol, Tbuf}
    push!(p.buffer, s)
    st = viewconsecutive(p.buffer, N, N)
    sz = size(st)
    p.policy(reshape(st, sz[1:end-2]..., sz[end-1] * sz[end]))
    
end
function defaultnmarkovpolicy(learner, buffer, π)
    if learner.nmarkov == 1
        π
    else
        NMarkovPolicy(learner.nmarkov, 
                      π, 
                      CircularArrayBuffer{eltype(buffer.states)}(learner.nmarkov, size(buffer.states)[1:end-1]))
    end
end

"""
    mutable struct ForcedPolicy 
        t::Int64
        actions::Array{Int64, 1}
"""
mutable struct ForcedPolicy 
    t::Int64
    actions::Array{Int64, 1}
end
ForcedPolicy(actions) = ForcedPolicy(1, actions)
function (p::ForcedPolicy)(s)
    if p.t == length(p.actions)
        p.t = 1
    else
        p.t += 1
    end
    p.actions[p.t]
end
