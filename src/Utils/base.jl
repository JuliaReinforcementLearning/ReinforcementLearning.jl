export findallmax, discount_rewards, discount_rewards!

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
    for i in (length(rewards)-1):-1:1
        new_rewards[i] = rewards[i] + new_rewards[i+1] * γ
    end
    new_rewards
end

discount_rewards(rewards, γ) = discount_rewards!(similar(rewards), rewards, γ)

discount_rewards_reduced(rewards, γ) = foldr((r, g)->r + γ * g, rewards)