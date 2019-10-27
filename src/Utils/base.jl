export findallmax,
       discount_rewards,
       discount_rewards!,
       CachedSampleAvg,
       SampleAvg,
       CachedSum,
       huber_loss,
       drop_last_dim

"""
    huber_loss(labels, predictions;δ = 1.0)

See [huber_loss](https://en.m.wikipedia.org/wiki/Huber_loss)
and the [implementation](https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/losses/losses_impl.py#L394-L469) in TensorFlow.

!!! warning
    The return is not reduced!
"""
function huber_loss(labels, predictions;δ = 1.0f0)
    abs_error = abs.(predictions .- labels)
    quadratic = min.(abs_error, 1.0f0)  # quadratic = min.(abs_error, δ)
    linear = abs_error .- quadratic
    0.5f0 .* quadratic .* quadratic .+ linear  # 0.5f0 .* quadratic .* quadratic .+ δ .* linear
end

"""
    findallmax(A::AbstractArray)

Like `findmax`, but all the indices of the maximum value are returned.

!!! warning
    All elements of value `NaN` in `A` will be ignored, unless all elements are `NaN`.
    In that case, the returned maximum value will be `NaN` and the returned indices will be `collect(1:length(A))`

#Examples
```julia-repl
julia> findallmax([-Inf, -Inf, -Inf])
(-Inf, [1, 2, 3])

julia> findallmax([Inf, Inf, Inf])
(Inf, [1, 2, 3])

julia> findallmax([Inf, 0, Inf])
(Inf, [1, 3])

julia> findallmax([0,1,2,1,2,1,0])
(2, [3, 5])
```
"""
function findallmax(A)
    maxval = typemin(eltype(A))
    idxs = Int[]
    for (i, x) in enumerate(A)
        if !isnan(x)
            if x > maxval
                maxval = x
                empty!(idxs)
                push!(idxs, i)
            elseif x == maxval
                push!(idxs, i)
            end
        end
    end
    if length(idxs) == 0
        NaN, collect(1:length(A))
    else
        maxval, idxs
    end
end

function discount_rewards!(new_rewards, rewards, γ)
    new_rewards[end] = rewards[end]
    for i = (length(rewards) - 1):-1:1
        new_rewards[i] = rewards[i] + new_rewards[i+1] * γ
    end
    new_rewards
end

discount_rewards(rewards, γ) = discount_rewards!(similar(rewards), rewards, γ)

discount_rewards_reduced(rewards, γ) = foldr((r, g) -> r + γ * g, rewards)

mutable struct SampleAvg
    t::Int
    avg::Float64
    SampleAvg() = new(0, 0)
end

function (s::SampleAvg)(x)
    s.t += 1
    s.avg += (x - s.avg) / s.t
    s.avg
end

struct CachedSampleAvg
    cache::Dict{Any,SampleAvg}
    CachedSampleAvg() = new(Dict())
end

function (c::CachedSampleAvg)(k, x)
    if !haskey(c.cache, k)
        c.cache[k] = SampleAvg()
    end
    c.cache[k](x)
end

struct CachedSum
    cache::Dict{Any,Float64}
    CachedSum() = new(Dict{Any,Float64}())
end

function (c::CachedSum)(k, x)
    c.cache[k] = get!(c.cache, k, 0.0) + x
end

drop_last_dim(x) = dropdims(x; dims=ndims(x))