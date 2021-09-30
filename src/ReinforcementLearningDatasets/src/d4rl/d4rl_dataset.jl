using Random
using HDF5

import Base: iterate, length, IteratorEltype

export dataset

export D4RLDataSet

"""
Represents an `Iterable` dataset with the following fields:

# Fields
- `dataset::Dict{Symbol, Any}`: representation of the dataset as a Dictionary with style as `style`.
- `repo::String`: the repository from which the dataset is taken.
- `dataset_size::Int`, the number of samples in the dataset.
- `batch_size::Int`: the size of the batches returned by `iterate`.
- `style::Tuple{Symbol}`: the style of the `Iterator` that is returned, check out: [`SARTS`](@ref), [`SART`](@ref) and [`SA`](@ref) for types supported out of the box.
- `rng<:AbstractRNG`.
- `meta::Dict`: the metadata provided along with the dataset.
- `is_shuffle::Bool`: determines if the batches returned by `iterate` are shuffled.
"""
struct D4RLDataSet{T<:AbstractRNG} <: RLDataSet
    dataset::Dict{Symbol, Any}
    repo::String
    dataset_size::Integer
    batch_size::Integer
    style::Tuple
    rng::T
    meta::Dict
    is_shuffle::Bool
end

# TO-DO: include other functionality like is_sequential, will be implemented soon
# TO-DO: enable the users providing their own paths to datasets if they already have it
# TO-DO: add additional env arg to do complete verify function
"""
    dataset(dataset; <keyword arguments>)

Create a dataset enclosed in a [`D4RLDataSet`](@ref) `Iterable` type. Contain other related metadata
for the `dataset` that is passed. The returned type is an infinite or a finite `Iterator` 
respectively depending upon whether `is_shuffle` is `true` or `false`. For more information regarding
the dataset, refer to [D4RL](https://github.com/rail-berkeley/d4rl). Check out d4rl_pybullet_dataset_params() or d4rl_dataset_params().

# Arguments

- `dataset::String`: name of the datset.
- `repo::String="d4rl"`: name of the repository of the dataset. can be "d4rl" or "d4rl-pybullet".
- `style::Tuple{Symbol}=SARTS`: the style of the `Iterator` that is returned. can be [`SARTS`](@ref), [`SART`](@ref) or [`SA`](@ref).
- `rng<:AbstractRNG=StableRNG(123)`.
- `is_shuffle::Bool=true`: determines if the dataset is shuffled or not.
- `batch_size::Int=256`: batch_size that is yielded by the iterator.

!!! note

[`FLOW`](https://flow-project.github.io/) and [`CARLA`](https://github.com/rail-berkeley/d4rl/wiki/CARLA-Setup) supported by [D4RL](https://github.com/rail-berkeley/d4rl) have not 
been tested in this package yet.
"""
function dataset(
    dataset::String;
    repo::String="d4rl",
    style::NTuple=SARTS,
    rng::AbstractRNG=MersenneTwister(123), 
    is_shuffle::Bool=true, 
    batch_size::Int=256
)
    
    try 
        @datadep_str repo*"-"*dataset 
    catch e
        if isa(e, KeyError)
            throw("Invalid params, check out d4rl_pybullet_dataset_params() or d4rl_dataset_params()")
        end
    end
        
    path = @datadep_str repo*"-"*dataset 

    @assert length(readdir(path)) == 1
    file_name = readdir(path)[1]
    
    data = h5open(path*"/"*file_name, "r") do file
        read(file)
    end

    # sanity checks on data
    d4rl_verify(data)

    dataset = Dict{Symbol, Any}()
    meta = Dict{String, Any}()

    N_samples = size(data["observations"])[2]
    
    for (key, d_key) in zip(["observations", "actions", "rewards", "terminals"], Symbol.(["state", "action", "reward", "terminal"]))
            dataset[d_key] = data[key]
    end
    
    for key in keys(data)
        if !(key in ["observations", "actions", "rewards", "terminals"])
            meta[key] = data[key]
        end
    end

    return D4RLDataSet(dataset, repo, N_samples, batch_size, style, rng, meta, is_shuffle)

end

function iterate(ds::D4RLDataSet, state = 0)
    rng = ds.rng
    batch_size = ds.batch_size
    size = ds.dataset_size
    is_shuffle = ds.is_shuffle
    style = ds.style

    if is_shuffle
        inds = rand(rng, 1:size-1, batch_size)
    else
        if (state+1) * batch_size <= size
            inds = state*batch_size+1:(state+1)*batch_size
        else
            return nothing
        end
        state += 1
    end

    batch = (state = ds.dataset[:state][:, inds],
    action = ds.dataset[:action][:, inds],
    reward = ds.dataset[:reward][inds],
    terminal = ds.dataset[:terminal][inds])

    if style == SARTS
        batch = merge(batch, (next_state = ds.dataset[:state][:, (1).+(inds)],))
    end
    
    return batch, state
end


take(ds::D4RLDataSet, n::Integer) = take(ds.dataset, n)
length(ds::D4RLDataSet) = ds.dataset_size
IteratorEltype(::Type{D4RLDataSet}) = EltypeUnknown() # see if eltype can be known (not sure about carla and adroit)


function d4rl_verify(data::Dict{String, Any})
    for key in ["observations", "actions", "rewards", "terminals"]
        @assert (key in keys(data)) "Expected keys not present in data"
    end
    N_samples = size(data["observations"])[2]
    @assert size(data["rewards"]) == (N_samples,) || size(data["rewards"]) == (1, N_samples)
    @assert size(data["terminals"]) == (N_samples,) || size(data["terminals"]) == (1, N_samples)
end
