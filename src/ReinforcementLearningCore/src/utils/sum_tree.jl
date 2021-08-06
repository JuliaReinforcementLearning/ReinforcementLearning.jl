export capacity, sample, SumTree

using Random
import StatsBase: sample

"""
    SumTree(capacity::Int)
Efficiently sample and update weights.
For more details, see the post at [here](https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/).
Here we use a vector to represent the binary tree.
Suppose we will have `capacity` leaves at most.
Every time we `push!` new node into the tree, only the recent `capacity` node and their sum will be updated!
[------------Parent nodes------------][--------leaves--------]
[size: 2^ceil(Int, log2(capacity))-1 ][     size: capacity   ]
# Example
```julia
julia> t = SumTree(8)
0-element SumTree
julia> for i in 1:16
       push!(t, i)
       end
julia> t
8-element SumTree:
  9.0
 10.0
 11.0
 12.0
 13.0
 14.0
 15.0
 16.0
julia> sample(t)
(2, 10.0)
julia> sample(t)
(1, 9.0)
julia> inds, ps = sample(t,100000)
([8, 4, 8, 1, 5, 2, 2, 7, 6, 6  …  1, 1, 7, 1, 6, 1, 5, 7, 2, 7], [16.0, 12.0, 16.0, 9.0, 13.0, 10.0, 10.0, 15.0, 14.0, 14.0  …  9.0, 9.0, 15.0, 9.0, 14.0, 9.0, 13.0, 15.0, 10.0, 15.0])
julia> countmap(inds)
Dict{Int64,Int64} with 8 entries:
  7 => 14991
  4 => 12019
  2 => 10003
  3 => 11027
  5 => 12971
  8 => 16052
  6 => 13952
  1 => 8985
julia> countmap(ps)
Dict{Float64,Int64} with 8 entries:
  9.0  => 8985
  13.0 => 12971
  10.0 => 10003
  14.0 => 13952
  16.0 => 16052
  11.0 => 11027
  15.0 => 14991
  12.0 => 12019
```
"""
mutable struct SumTree{T} <: AbstractArray{Int,1}
    capacity::Int
    first::Int
    length::Int
    nparents::Int
    tree::Vector{T}
    SumTree(capacity::Int) = SumTree(Float32, capacity)
    function SumTree(T, capacity)
        nparents = 2^ceil(Int, log2(capacity)) - 1
        new{T}(capacity, 1, 0, nparents, zeros(T, nparents + capacity))
    end
end

capacity(t::SumTree) = t.capacity
Base.length(t::SumTree) = t.length
Base.size(t::SumTree) = (length(t),)
Base.eltype(t::SumTree{T}) where {T} = T

function _index(t::SumTree, i::Int)
    ind = i + t.first - 1
    if ind > t.capacity
        ind -= t.capacity
    end
    ind
end

_tree_index(t::SumTree, i) = t.nparents + _index(t, i)

Base.getindex(t::SumTree, i::Int) = t.tree[_tree_index(t, i)]

function Base.setindex!(t::SumTree, p, i)
    tree_ind = _tree_index(t, i)
    change = p - t.tree[tree_ind]
    t.tree[tree_ind] = p
    while tree_ind != 1
        tree_ind = tree_ind ÷ 2
        t.tree[tree_ind] += change
    end
end

function Base.push!(t::SumTree, p)
    if t.length == t.capacity
        t.first = (t.first == t.capacity ? 1 : t.first + 1)
    else
        t.length += 1
    end
    t[t.length] = p
end

function Base.pop!(t::SumTree)
    if t.length > 0
        res = t[end]
        t.length -= 1
        res
    else
        @error "can not pop! from an empty SumTree"
    end
end

function Base.empty!(t::SumTree)
    t.length = 0.0
    fill!(t.tree, 0.0)
    # yes, no need to reset `t.first`
    # so, don't rely on that `t.first` is always 1 after `empty!`
    t
end

function Base.get(t::SumTree, v)
    parent_ind = 1
    leaf_ind = parent_ind
    while true
        left_child_ind = parent_ind * 2
        right_child_ind = left_child_ind + 1
        if left_child_ind > length(t.tree)
            leaf_ind = parent_ind
            break
        else
            if v ≤ t.tree[left_child_ind]
                parent_ind = left_child_ind
            else
                v -= t.tree[left_child_ind]
                parent_ind = right_child_ind
            end
        end
    end
    if leaf_ind <= t.nparents
        leaf_ind += t.capacity
    end
    p = t.tree[leaf_ind]
    ind = leaf_ind - t.nparents
    real_ind = ind >= t.first ? ind - t.first + 1 : ind + t.capacity - t.first + 1
    real_ind, p
end

sample(rng::AbstractRNG, t::SumTree{T}) where {T} = get(t, rand(rng, T) * t.tree[1])
sample(t::SumTree) = sample(Random.GLOBAL_RNG, t)

function sample(rng::AbstractRNG, t::SumTree{T}, n::Int) where {T}
    inds, priorities = Vector{Int}(undef, n), Vector{Float64}(undef, n)
    for i in 1:n
        v = (i - 1 + rand(rng, T)) / n
        ind, p = get(t, v * t.tree[1])
        inds[i] = ind
        priorities[i] = p
    end
    inds, priorities
end

sample(t::SumTree, n::Int) = sample(Random.GLOBAL_RNG, t, n)
