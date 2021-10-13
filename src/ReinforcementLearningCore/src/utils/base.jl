export nframes,
    select_last_dim,
    select_last_frame,
    consecutive_view,
    find_all_max,
    discount_rewards,
    discount_rewards!,
    discount_rewards_reduced,
    generalized_advantage_estimation,
    generalized_advantage_estimation!,
    flatten_batch

using StatsBase
using Compat

nframes(a::AbstractArray{T,N}) where {T,N} = size(a, N)

select_last_dim(xs::AbstractArray{T,N}, inds) where {T,N} =
    @views xs[ntuple(_ -> (:), N - 1)..., inds]

select_last_frame(xs::AbstractArray{T,N}) where {T,N} = select_last_dim(xs, size(xs, N))

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

"""
    consecutive_view(x::AbstractArray, inds; n_stack = nothing, n_horizon = nothing)

By default, it behaves the same with `select_last_dim(x, inds)`.
If `n_stack` is set to an int, then for each frame specified by `inds`,
the previous `n_stack` frames (including the current one) are concatenated as a new dimension.
If `n_horizon` is set to an int, then for each frame specified by `inds`,
the next `n_horizon` frames (including the current one) are concatenated as a new dimension.

# Example

```julia
julia> x = collect(1:5)
5-element Array{Int64,1}:
 1
 2
 3
 4
 5

julia> consecutive_view(x, [2,4])  # just the same with `select_last_dim(x, [2,4])`
2-element view(::Array{Int64,1}, [2, 4]) with eltype Int64:
 2
 4

julia> consecutive_view(x, [2,4];n_stack = 2)
2×2 view(::Array{Int64,1}, [1 3; 2 4]) with eltype Int64:
 1  3
 2  4

julia> consecutive_view(x, [2,4];n_horizon = 2)
2×2 view(::Array{Int64,1}, [2 4; 3 5]) with eltype Int64:
 2  4
 3  5

julia> consecutive_view(x, [2,4];n_horizon = 2, n_stack=2)  # note the order here, first we stack, then we apply the horizon
2×2×2 view(::Array{Int64,1}, [1 2; 2 3]

[3 4; 4 5]) with eltype Int64:
[:, :, 1] =
 1  2
 2  3

[:, :, 2] =
 3  4
 4  5
```

See also [Frame Skipping and Preprocessing for Deep Q networks](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/)
to gain a better understanding of state stacking and n-step learning.
"""
consecutive_view(
    cb::AbstractArray,
    inds::Vector{Int};
    n_stack = nothing,
    n_horizon = nothing,
) = consecutive_view(cb, inds, n_stack, n_horizon)

consecutive_view(cb::AbstractArray, inds::Vector{Int}, ::Nothing, ::Nothing) =
    select_last_dim(cb, inds)

consecutive_view(cb::AbstractArray, inds::Vector{Int}, n_stack::Int, ::Nothing) =
    select_last_dim(
        cb,
        reshape([i for x in inds for i in x-n_stack+1:x], n_stack, length(inds)),
    )

consecutive_view(cb::AbstractArray, inds::Vector{Int}, ::Nothing, n_horizon::Int) =
    select_last_dim(
        cb,
        reshape([i for x in inds for i in x:x+n_horizon-1], n_horizon, length(inds)),
    )

consecutive_view(cb::AbstractArray, inds::Vector{Int}, n_stack::Int, n_horizon::Int) =
    select_last_dim(
        cb,
        reshape(
            [j for x in inds for i in x:x+n_horizon-1 for j in i-n_stack+1:i],
            n_stack,
            n_horizon,
            length(inds),
        ),
    )

function find_all_max(x)
    v = maximum(x)
    v, findall(==(v), x)
end

function find_all_max(x, mask::AbstractVector{Bool})
    v = maximum(view(x, mask))
    v, [k for (m, k) in zip(mask, keys(x)) if m && x[k] == v]
end

# !!! watch https://github.com/JuliaLang/julia/pull/35316#issuecomment-622629895
# Base.findmax(f, domain) = mapfoldl(x -> (f(x), x), _rf_findmax, domain)
# _rf_findmax((fm, m), (fx, x)) = isless(fm, fx) ? (fx, x) : (fm, m)

# !!! type piracy
Base.findmax(A::AbstractVector{T}, mask::AbstractVector{Bool}) where T = findmax(ifelse.(mask, A, typemin(T)))


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

discount_rewards!(new_rewards, rewards, γ; terminal = nothing, init = nothing, dims = :) =
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
    for (r′, r) in zip(eachslice(new_rewards, dims = dims), eachslice(rewards, dims = dims))
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
        enumerate(zip(eachslice(new_rewards, dims = dims), eachslice(rewards, dims = dims)))
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
        eachslice(new_rewards, dims = dims),
        eachslice(rewards, dims = dims),
        eachslice(terminal, dims = dims),
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
            eachslice(new_rewards, dims = dims),
            eachslice(rewards, dims = dims),
            eachslice(terminal, dims = dims),
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

discount_rewards_reduced(rewards::AbstractVector, γ; terminal = nothing, init = nothing) =
    _discount_rewards_reduced(rewards, γ, terminal, init)

function discount_rewards_reduced(
    rewards::AbstractMatrix,
    γ::T;
    terminal = nothing,
    init = nothing,
    dims,
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
    terminal = nothing,
    init = nothing,
    dims,
) = _discount_rewards_reduced!(reduced_rewards, rewards, γ, terminal, init, dims)

function _discount_rewards_reduced!(
    reduced_rewards,
    rewards,
    γ,
    terminal::Nothing,
    init::Nothing,
    dims::Int,
)
    for (i, r) in enumerate(eachslice(rewards, dims = dims))
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
    for (i, r) in enumerate(eachslice(rewards, dims = dims))
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
        enumerate(zip(eachslice(rewards, dims = dims), eachslice(terminal, dims = dims)))
        reduced_rewards[i] = _discount_rewards_reduced(r, γ, t, nothing)
    end
end

function _discount_rewards_reduced!(reduced_rewards, rewards, γ, terminal, init, dims::Int)
    for (i, (r, t)) in
        enumerate(zip(eachslice(rewards, dims = dims), eachslice(terminal, dims = dims)))
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
    kwargs...,
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
    terminal = nothing,
    dims = :,
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
        eachslice(advantages, dims = dims),
        eachslice(rewards, dims = dims),
        eachslice(values, dims = dims),
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
        eachslice(advantages, dims = dims),
        eachslice(rewards, dims = dims),
        eachslice(values, dims = dims),
        eachslice(terminal, dims = dims),
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
