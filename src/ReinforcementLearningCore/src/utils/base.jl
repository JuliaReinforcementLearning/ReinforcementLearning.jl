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
    logitcrossentropy_unreduced

using StatsBase

select_last_dim(xs::AbstractArray{T,N}, inds) where {T,N} =
    @views xs[ntuple(_ -> (:), N - 1)..., inds]
select_last_frame(xs::AbstractArray{T,N}) where {T,N} = select_last_dim(xs, size(xs, N))


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

function huber_loss_unreduced(labels, predictions; δ = 1.0f0)
    abs_error = abs.(predictions .- labels)
    quadratic = min.(abs_error, 1.0f0)  # quadratic = min.(abs_error, δ)
    linear = abs_error .- quadratic
    0.5f0 .* quadratic .* quadratic .+ linear  # 0.5f0 .* quadratic .* quadratic .+ δ .* linear
end

huber_loss(labels, predictions; δ = 1.0f0) =
    huber_loss_unreduced(labels, predictions; δ = δ) |> mean

function discount_rewards!(new_rewards, rewards, γ)
    new_rewards[end] = rewards[end]
    for i in (length(rewards)-1):-1:1
        new_rewards[i] = rewards[i] + new_rewards[i+1] * γ
    end
    new_rewards
end

discount_rewards(rewards, γ) = discount_rewards!(similar(rewards), rewards, γ)

discount_rewards_reduced(rewards, γ) = foldr((r, g) -> r + γ * g, rewards)
