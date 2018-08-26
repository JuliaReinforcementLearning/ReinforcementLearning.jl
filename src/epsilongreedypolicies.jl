"""
    mutable struct EpsilonGreedyPolicy{kind}
        ϵ::Float64

Chooses the action with the highest value with probability `1 - ϵ` and selects 
an action uniformly random with probability `ϵ`.
"""
mutable struct EpsilonGreedyPolicy{kind}
    ϵ::Float64
end
EpsilonGreedyPolicy(ϵ; kind = :veryoptimistic) = EpsilonGreedyPolicy{kind}(ϵ)
export EpsilonGreedyPolicy

import Base.maximum, Base.isequal
maximum(::EpsilonGreedyPolicy, v) = maximumbelowInf(v)
isequal(::EpsilonGreedyPolicy{:optimistic}, v1, v2) = v1 >= v2
maximum(::EpsilonGreedyPolicy{:veryoptimistic}, v) = maximum(v)
isequal(::EpsilonGreedyPolicy, v1, v2) = v1 == v2

function selectaction(policy::EpsilonGreedyPolicy, na, f, input)
    if rand() < policy.ϵ
        rand(1:na)
    else
        values = f(input)
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
end
selectaction(policy::EpsilonGreedyPolicy, values) = 
    selectaction(policy, length(values), x -> x, values)

function getactionprobabilities(policy::EpsilonGreedyPolicy, values)
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

