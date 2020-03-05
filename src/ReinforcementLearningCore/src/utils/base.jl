export select_last_dim,
    select_last_frame,
    consecutive_view,
    find_all_max,
    find_max,
    huber_loss,
    huber_loss_unreduced,
    discount_rewards,
    discount_rewards!,
    discount_rewards_reduced,
    logitcrossentropy_unreduced,
    flatten_batch,
    unflatten_batch

using StatsBase

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
flatten_batch(x::AbstractArray) =
    reshape(x, (size(x) |> reverse |> Base.tail |> Base.tail |> reverse)..., :)  # much faster than  `reshape(x, size(x)[1:end-2]..., :)`

unflatten_batch(x::AbstractArray, i::Int...) =
    reshape(x, (size(x) |> reverse |> Base.tail |> reverse)..., i...)

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
consecutive_view(cb::AbstractArray, inds::Vector{Int}, ::Nothing, n_horizeon::Int) =
    select_last_dim(
        cb,
        reshape([i for x in inds for i in x:x+n_horizeon-1], n_horizeon, length(inds)),
    )
consecutive_view(cb::AbstractArray, inds::Vector{Int}, n_stack::Int, n_horizeon::Int) =
    select_last_dim(
        cb,
        reshape(
            [j for x in inds for i in x:x+n_horizeon-1 for j in i-n_stack+1:i],
            n_stack,
            n_horizeon,
            length(inds),
        ),
    )

"""
    find_all_max(A::AbstractArray)

Like `find_max`, but all the indices of the maximum value are returned.

!!! warning
    All elements of value `NaN` in `A` will be ignored, unless all elements are `NaN`.
    In that case, the returned maximum value will be `NaN` and the returned indices will be `collect(1:length(A))`

# Examples

```julia-repl
julia> find_all_max([-Inf, -Inf, -Inf])
(-Inf, [1, 2, 3])
julia> find_all_max([Inf, Inf, Inf])
(Inf, [1, 2, 3])
julia> find_all_max([Inf, 0, Inf])
(Inf, [1, 3])
julia> find_all_max([0,1,2,1,2,1,0])
(2, [3, 5])
```
"""
function find_all_max(A)
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

"""
    find_all_max(A, mask)

Similar to `find_all_max(A)`, but only the masked elements in `A` will be considered.
"""
function find_all_max(A, mask)
    maxval = typemin(eltype(A))
    idxs = Int[]
    for (i, x) in enumerate(A)
        if mask[i] && (!isnan(x))
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

find_max(A) = findmax(A)

function find_max(A, mask)
    maxval = typemin(eltype(A))
    ind = 0
    for (i, x) in enumerate(A)
        if mask[i] && x >= maxval
            maxval = x
            ind = i
        end
    end
    maxval, ind
end

function logitcrossentropy_unreduced(logŷ::AbstractVecOrMat, y::AbstractVecOrMat)
    return vec(-sum(y .* logsoftmax(logŷ), dims = 1))
end

"""
    huber_loss_unreduced(labels, predictions; δ = 1.0f0)

Similar to [`huber_loss`](@ref), but it doesn't do the `mean` operation in the last step.
"""
function huber_loss_unreduced(labels, predictions; δ = 1.0f0)
    abs_error = abs.(predictions .- labels)
    quadratic = min.(abs_error, 1.0f0)  # quadratic = min.(abs_error, δ)
    linear = abs_error .- quadratic
    0.5f0 .* quadratic .* quadratic .+ linear  # 0.5f0 .* quadratic .* quadratic .+ δ .* linear
end

"""
    huber_loss(labels, predictions; δ = 1.0f0)

See [Huber loss](https://en.wikipedia.org/wiki/Huber_loss)
"""
huber_loss(labels, predictions; δ = 1.0f0) =
    huber_loss_unreduced(labels, predictions; δ = δ) |> mean

const VectorOrMatrix = Union{AbstractMatrix,AbstractVector}

"""
    discount_rewards(rewards::VectorOrMatrix, γ::Number;kwargs...)

Calculate the gain started from the current step with discount rate of `γ`.
`rewards` can be a matrix.

# Keyword argments

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
    for (i, (r′, r, t)) in enumerate(zip(
        eachslice(new_rewards, dims = dims),
        eachslice(rewards, dims = dims),
        eachslice(terminal, dims = dims),
    ))
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
