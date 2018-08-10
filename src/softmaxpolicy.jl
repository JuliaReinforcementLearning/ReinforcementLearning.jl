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
abstract type AbstractSoftmaxPolicy end
mutable struct SoftmaxPolicy <: AbstractSoftmaxPolicy
    β::Float64
end
SoftmaxPolicy(; β = 1.) = SoftmaxPolicy(β)
export SoftmaxPolicy

function selectaction(policy::AbstractSoftmaxPolicy, values)
    if maximum(values) == Inf64
        rand(findall(v -> v == Inf64, values))
    else
        actsoftmax(policy, values)
    end
end

function getactionprobabilities(policy::AbstractSoftmaxPolicy, values)
    if maximum(values) == Inf64
        p = zeros(length(values))
        a = findall(v -> v == Inf64, values)
        for i in a
            p[i] = 1/length(a)
        end
        return p
    else
        expvals = getexpvals(policy, values)
        return expvals/sum(expvals)
    end
end

@inline getexpvals(p::SoftmaxPolicy, values) = exp.(p.β .* (values .- maximum(values)))

# Samples from Categorical(exp(input)/sum(exp(input)))
function actsoftmax(policy::SoftmaxPolicy, values)
    if policy.β == Inf
        argmax(values)
    else
        actsoftmax(policy.β .* values)
    end
end
function actsoftmax(input)
    unnormalized_probs = exp.(input)
    r = rand()*sum(unnormalized_probs)
    tmp = unnormalized_probs[1]
    @inbounds for i = 1:length(unnormalized_probs) - 1
        if  r <= tmp
            return i
        end
        tmp += unnormalized_probs[i + 1]
    end
    return length(unnormalized_probs)
end

