using Distributions: DiscreteDistribution, ContinuousDistribution
using Flux: softmax

export action_distribution

"""
    action_distribution(dist, model_output)

Compute the action distribution using the distribution type and output from a model.
"""
action_distribution(dist, model_output) =
    throw(ArgumentError("dist ($dist) is not a ContinuousDistribution or DiscreteDistribution, not implemented"))

"""
    action_distribution(dist::Type{T}, model_output) where {T<:DiscreteDistribution}

When `dist` is a subtype of `DiscreteDistribution`, assume `model_output` is a batch of unnomalized log probabilities.
# Examples
```jldoctest
julia> model_output = reshape(1:10, 5, 2)
5×2 reshape(::UnitRange{Int64}, 5, 2) with eltype Int64:
 1   6
 2   7
 3   8
 4   9
 5  10
julia> action_distribution(Categorical, model_output)
2-element Vector{Categorical{Float64, Vector{Float64}}}:
 Categorical{Float64, Vector{Float64}}(
support: Base.OneTo(5)
p: [0.011656230956039605, 0.03168492079612427, 0.0861285444362687, 0.23412165725273662, 0.6364086465588308]
)

 Categorical{Float64, Vector{Float64}}(
support: Base.OneTo(5)
p: [0.011656230956039605, 0.03168492079612427, 0.0861285444362687, 0.23412165725273662, 0.6364086465588308]
)
```
"""
action_distribution(dist::Type{T}, model_output) where {T<:DiscreteDistribution} = 
    map(col -> dist(col, check_args=false), eachcol(softmax(model_output)))

"""
    action_distribution(dist::Type{T}, model_output) where {T<:ContinuousDistribution}

When `dist` is a subtype of `ContinuousDistribution`, assume `model_output` are a batch of parameters to be supplied to `dist`.
# Examples
```jldoctest
julia> model_output = reshape(1:10, 2, 5)
2×5 reshape(::UnitRange{Int64}, 2, 5) with eltype Int64:
 1  3  5  7   9
 2  4  6  8  10
julia> action_distribution(Normal, model_output)
5-element Vector{Normal{Float64}}:
 Normal{Float64}(μ=1.0, σ=2.0)
 Normal{Float64}(μ=3.0, σ=4.0)
 Normal{Float64}(μ=5.0, σ=6.0)
 Normal{Float64}(μ=7.0, σ=8.0)
 Normal{Float64}(μ=9.0, σ=10.0)
```
"""
action_distribution(dist::Type{T}, model_output) where {T<:ContinuousDistribution} = 
    map(col -> dist(Vector(col)), eachcol(model_output))