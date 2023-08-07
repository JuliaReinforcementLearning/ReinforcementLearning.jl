export global_norm,
    clip_by_global_norm!,
    find_all_max,
    discount_rewards,
    discount_rewards!,
    discount_rewards_reduced,
    generalized_advantage_estimation,
    generalized_advantage_estimation!,
    flatten_batch,
    orthogonal

using FillArrays: Trues
using GPUArrays

#####
# Zygote
#####

global_norm(gs, ps) = sqrt(sum(mapreduce(x -> x^2, +, gs[p]) for p in ps))

function clip_by_global_norm!(gs, ps, clip_norm::Float32)
    gn = global_norm(gs, ps)
    if clip_norm <= gn
        for p in ps
            gs[p] .*= clip_norm / max(clip_norm, gn)
        end
    end
    gn
end

#####
# Flux
#####

# https://github.com/FluxML/Flux.jl/pull/1171/
# https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Orthogonal
function orthogonal_matrix(rng::AbstractRNG, nrow, ncol)
    shape = reverse(minmax(nrow, ncol))
    a = randn(rng, Float32, shape)
    q, r = qr(a)
    q = Matrix(q) * diagm(sign.(diag(r)))
    nrow < ncol ? permutedims(q) : q
end

function orthogonal(rng::AbstractRNG, d1, rest_dims...)
    m = orthogonal_matrix(rng, d1, *(rest_dims...))
    reshape(m, d1, rest_dims...)
end

orthogonal(dims...) = orthogonal(Random.default_rng(), dims...)
orthogonal(rng::AbstractRNG) = (dims...) -> orthogonal(rng, dims...)

#####
# MLUtils
#####

"""
    flatten_batch(x::AbstractArray)

Merge the last two dimension.

# Example

```julia-repl
julia> x = reshape(1:12, 2, 2, 3)
2×2×3 reshape(::UnitRange{Int64}, 2, 2, 3) with eltype Int64:
[:, :, 1] =
 1  3
 2  4

[:, :, 2] =
 5  7
 6  8

[:, :, 3] =
  9  11
 10  12

julia> flatten_batch(x)
2×6 reshape(::UnitRange{Int64}, 2, 6) with eltype Int64:
 1  3  5  7   9  11
 2  4  6  8  10  12
```
"""
flatten_batch(x::AbstractArray) = reshape(x, size(x)[1:end-2]..., :)

#####
# RLUtils
#####

function find_all_max(x::A) where {A <: AbstractArray}
    v = maximum(x)
    indices = Vector{Int}(undef, count(==(v), x))
    j = 1
    for i in eachindex(x)
        if x[i] == v
            indices[j] = i
            j += 1
        end
    end
    v, indices
end

function find_all_max(x::A) where {A <: AbstractGPUArray}
    v = maximum(x)
    v, findall(==(v), x)
end

find_all_max(x, mask::Trues) = find_all_max(x)

function find_all_max(x, mask::AbstractVector{Bool})
    v = maximum(view(x, mask))
    v, [k for (m, k) in zip(mask, keys(x)) if m && x[k] == v]
end

# !!! watch https://github.com/JuliaLang/julia/pull/35316#issuecomment-622629895
# Base.findmax(f, domain) = mapfoldl(x -> (f(x), x), _rf_findmax, domain)
# _rf_findmax((fm, m), (fx, x)) = isless(fm, fx) ? (fx, x) : (fm, m)

# !!! type piracy
Base.findmax(A::AbstractVector{T}, mask::AbstractVector{Bool}) where {T} =
    findmax(ifelse.(mask, A, typemin(T)))

Base.findmax(A::AbstractVector, mask::Trues) = findmax(A)


const VectorOrMatrix = Union{AbstractMatrix,AbstractVector}

"""
    discount_rewards(rewards::VectorOrMatrix, γ::Number;kwargs...)

Calculate the gain started from the current step with discount rate of `γ`.
`rewards` can be a matrix.

# Keyword arguments

- `dims=:`, if `rewards` is a `Matrix`, then `dims` can only be `1` or `2`.
- `terminal=nothing`, specify if each reward follows by a terminal. `nothing` means the game is not terminated yet. If `terminal` is provided, then the size must be the same with `rewards`.
- `init=nothing`, `init` can be used to provide the the reward estimation of the last state.

# Example
"""
function discount_rewards(rewards::VectorOrMatrix, γ::T; kwargs...) where {T<:Number}
    res = similar(rewards, promote_type(eltype(rewards), T))
    discount_rewards!(res, rewards, γ; kwargs...)
    res
end

discount_rewards!(new_rewards, rewards, γ; terminal=nothing, init=nothing, dims=:) =
    _discount_rewards!(new_rewards, rewards, γ, terminal, init, dims)

function _discount_rewards!(
    new_rewards::AbstractMatrix,
    rewards::AbstractMatrix,
    γ,
    terminal::Nothing,
    init::Nothing,
    dims::Int,
)
    dims = ndims(rewards) - dims + 1
    for (r′, r) in zip(eachslice(new_rewards, dims=dims), eachslice(rewards, dims=dims))
        _discount_rewards!(r′, r, γ, nothing, nothing)
    end
end

function _discount_rewards!(
    new_rewards::AbstractMatrix,
    rewards::AbstractMatrix,
    γ,
    terminal::Nothing,
    init,
    dims::Int,
)
    dims = ndims(rewards) - dims + 1
    for (i, (r′, r)) in
        enumerate(zip(eachslice(new_rewards, dims=dims), eachslice(rewards, dims=dims)))
        _discount_rewards!(r′, r, γ, nothing, init[i])
    end
end

function _discount_rewards!(
    new_rewards::AbstractMatrix,
    rewards::AbstractMatrix,
    γ,
    terminal,
    init::Nothing,
    dims::Int,
)
    dims = ndims(rewards) - dims + 1
    for (r′, r, t) in zip(
        eachslice(new_rewards, dims=dims),
        eachslice(rewards, dims=dims),
        eachslice(terminal, dims=dims),
    )
        _discount_rewards!(r′, r, γ, t, nothing)
    end
end

function _discount_rewards!(
    new_rewards::AbstractMatrix,
    rewards::AbstractMatrix,
    γ,
    terminal,
    init,
    dims::Int,
)
    dims = ndims(rewards) - dims + 1
    for (i, (r′, r, t)) in enumerate(
        zip(
            eachslice(new_rewards, dims=dims),
            eachslice(rewards, dims=dims),
            eachslice(terminal, dims=dims),
        ),
    )
        _discount_rewards!(r′, r, γ, t, init[i])
    end
end

_discount_rewards!(
    new_rewards::AbstractVector,
    rewards::AbstractVector,
    γ,
    terminal,
    init,
    dims::Colon,
) = _discount_rewards!(new_rewards, rewards, γ, terminal, init)

"assuming rewards and new_rewards are Vector"
_discount_rewards!(new_rewards, rewards, γ, terminal, init::Nothing) =
    _discount_rewards!(new_rewards, rewards, γ, terminal, zero(eltype(new_rewards)))

function _discount_rewards!(new_rewards, rewards, γ, terminal, init)
    gain = init
    for i in length(rewards):-1:1
        is_continue = isnothing(terminal) ? true : (!terminal[i])
        gain = rewards[i] + γ * gain * is_continue
        new_rewards[i] = gain
    end
    new_rewards
end

discount_rewards_reduced(rewards::AbstractVector, γ; terminal=nothing, init=nothing) =
    _discount_rewards_reduced(rewards, γ, terminal, init)

function discount_rewards_reduced(
    rewards::AbstractMatrix,
    γ::T;
    terminal=nothing,
    init=nothing,
    dims
) where {T<:Number}
    dims = ndims(rewards) - dims + 1
    res = Array{promote_type(eltype(rewards), T)}(undef, size(rewards, dims))
    _discount_rewards_reduced!(res, rewards, γ, terminal, init, dims)
    res
end

_discount_rewards_reduced(rewards, γ, terminal, init::Nothing) =
    _discount_rewards_reduced(rewards, γ, terminal, zero(eltype(rewards)))

function _discount_rewards_reduced(rewards, γ, terminal, init)
    gain = init
    for i in length(rewards):-1:1
        is_continue = isnothing(terminal) ? true : (!terminal[i])
        gain = rewards[i] + γ * gain * is_continue
    end
    gain
end

discount_rewards_reduced!(
    reduced_rewards::AbstractVector,
    rewards::AbstractMatrix,
    γ;
    terminal=nothing,
    init=nothing,
    dims
) = _discount_rewards_reduced!(reduced_rewards, rewards, γ, terminal, init, dims)

function _discount_rewards_reduced!(
    reduced_rewards,
    rewards,
    γ,
    terminal::Nothing,
    init::Nothing,
    dims::Int,
)
    for (i, r) in enumerate(eachslice(rewards, dims=dims))
        reduced_rewards[i] = _discount_rewards_reduced(r, γ, nothing, nothing)
    end
end

function _discount_rewards_reduced!(
    reduced_rewards,
    rewards,
    γ,
    terminal::Nothing,
    init,
    dims::Int,
)
    for (i, r) in enumerate(eachslice(rewards, dims=dims))
        reduced_rewards[i] = _discount_rewards_reduced(r, γ, nothing, init[i])
    end
end

function _discount_rewards_reduced!(
    reduced_rewards,
    rewards,
    γ,
    terminal,
    init::Nothing,
    dims::Int,
)
    for (i, (r, t)) in
        enumerate(zip(eachslice(rewards, dims=dims), eachslice(terminal, dims=dims)))
        reduced_rewards[i] = _discount_rewards_reduced(r, γ, t, nothing)
    end
end

function _discount_rewards_reduced!(reduced_rewards, rewards, γ, terminal, init, dims::Int)
    for (i, (r, t)) in
        enumerate(zip(eachslice(rewards, dims=dims), eachslice(terminal, dims=dims)))
        reduced_rewards[i] = _discount_rewards_reduced(r, γ, t, init[i])
    end
end

"""
    generalized_advantage_estimation(rewards::VectorOrMatrix, values::VectorOrMatrix, γ::Number, λ::Number;kwargs...)

Calculate the generalized advantage estimate started from the current step with discount rate of `γ` and a lambda for GAE-Lambda of 'λ'.
`rewards` and 'values' can be a matrix.

# Keyword arguments

- `dims=:`, if `rewards` is a `Matrix`, then `dims` can only be `1` or `2`.
- `terminal=nothing`, specify if each reward follows by a terminal. `nothing` means the game is not terminated yet. If `terminal` is provided, then the size must be the same with `rewards`.

# Example
"""
function generalized_advantage_estimation(
    rewards::VectorOrMatrix,
    values::VectorOrMatrix,
    γ::T,
    λ::T;
    kwargs...
) where {T<:Number}
    res = similar(rewards, promote_type(eltype(rewards), T))
    generalized_advantage_estimation!(res, rewards, values, γ, λ; kwargs...)
    res
end

generalized_advantage_estimation!(
    advantages,
    rewards,
    values,
    γ,
    λ;
    terminal=nothing,
    dims=:
) = _generalized_advantage_estimation!(advantages, rewards, values, γ, λ, terminal, dims)

function _generalized_advantage_estimation!(
    advantages::AbstractMatrix,
    rewards::AbstractMatrix,
    values::AbstractMatrix,
    γ,
    λ,
    terminal::Nothing,
    dims::Int,
)
    dims = ndims(rewards) - dims + 1
    for (r′, r, v) in zip(
        eachslice(advantages, dims=dims),
        eachslice(rewards, dims=dims),
        eachslice(values, dims=dims),
    )
        _generalized_advantage_estimation!(r′, r, v, γ, λ, nothing)
    end
end


function _generalized_advantage_estimation!(
    advantages::AbstractMatrix,
    rewards::AbstractMatrix,
    values::AbstractMatrix,
    γ,
    λ,
    terminal,
    dims::Int,
)
    dims = ndims(rewards) - dims + 1
    for (r′, r, v, t) in zip(
        eachslice(advantages, dims=dims),
        eachslice(rewards, dims=dims),
        eachslice(values, dims=dims),
        eachslice(terminal, dims=dims),
    )
        _generalized_advantage_estimation!(r′, r, v, γ, λ, t)
    end
end

_generalized_advantage_estimation!(
    advantages::AbstractVector,
    rewards::AbstractVector,
    values::AbstractVector,
    γ,
    λ,
    terminal,
    dims::Colon,
) = _generalized_advantage_estimation!(advantages, rewards, values, γ, λ, terminal)


"assuming rewards and advantages are Vector"
function _generalized_advantage_estimation!(advantages, rewards, values, γ, λ, terminal)
    gae = 0
    for i in length(rewards):-1:1
        is_continue = isnothing(terminal) ? true : (!terminal[i])
        delta = rewards[i] + γ * values[i+1] * is_continue - values[i]
        gae = delta + γ * λ * is_continue * gae
        advantages[i] = gae
    end
    advantages
end
