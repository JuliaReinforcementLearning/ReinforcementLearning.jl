using Random
using StableRNGs
using HDF5

import Base: iterate, length, IteratorEltype

export dataset

export D4RLDataSet

"""
Represents an iterable dataset of type D4RLDataSet with the following fields:

`dataset`: Dict{Symbol, Any}, representation of the dataset as a Dictionary with style as `style`
`repo`: String, the repository from which the dataset is taken
`size`: Integer, the size of the dataset
`batch_size`: Integer, the size of the batches returned by `iterate`.
`style`: Tuple, the type of the NamedTuple, for now SARTS and SART is supported.
`rng`<: AbstractRNG.
`meta`: Dict, the metadata provided along with the dataset
`is_shuffle`: Bool, determines if the batches returned by `iterate` are shuffled.
"""
struct D4RLDataSet{T<:AbstractRNG} <: RLDataSet
    dataset::Dict{Symbol, Any}
    repo::String
    size::Integer
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
    dataset(dataset::String; style::Tuple, rng<:AbstractRNG, is_shuffle::Bool, max_iters::Int64, batch_size::Int64)

Creates a dataset of enclosed in a D4RLDataSet type and other related metadata for the `dataset` that is passed.
The `D4RLDataSet` type is an iterable that fetches batches when used in a for loop for convenience during offline training.

`dataset`: Dict{Symbol, Any}, Name of the datset.
`repo`: Name of the repository of the dataset.
`style`: the style of the iterator and the Dict inside D4RLDataSet that is returned.
`rng`: StableRNG
`max_iters`: maximum number of iterations for the iterator.
`is_shuffle`: whether the dataset is shuffled or not. `true` by default.
`batch_size`: batch_size that is yielded by the iterator. Defaults to 256.

The returned type is an infinite iterator which can be called using `iterate` and will return batches as specified in the dataset.
"""
function dataset(dataset::String;
    style=SARTS,
    repo = "d4rl",
    rng = StableRNG(123), 
    is_shuffle = true, 
    batch_size=256
)
    
    try 
        @datadep_str repo*"-"*dataset 
    catch 
        throw("The provided dataset is not available") 
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
    size = ds.size
    is_shuffle = ds.is_shuffle
    style = ds.style

    if is_shuffle
        inds = rand(rng, 1:size, batch_size)
        map((x)-> if x <= size x else 1 end, inds)
    else
        if (state+1) * batch_size <= size
            inds = state*batch_size+1:(state+1)*batch_size
        else
            return nothing
        end
        state += 1
    end

    batch = (state = copy(ds.dataset[:state][:, inds]),
    action = copy(ds.dataset[:action][:, inds]),
    reward = copy(ds.dataset[:reward][inds]),
    terminal = copy(ds.dataset[:terminal][inds]))

    if style == SARTS
        batch = merge(batch, (next_state = copy(ds.dataset[:state][:, (1).+(inds)]),))
    end
    
    return batch, state
end


take(ds::D4RLDataSet, n::Integer) = take(ds.dataset, n)
length(ds::D4RLDataSet) = ds.size
IteratorEltype(::Type{D4RLDataSet}) = EltypeUnknown() # see if eltype can be known (not sure about carla and adroit)


function d4rl_verify(data::Dict{String, Any})
    for key in ["observations", "actions", "rewards", "terminals"]
        @assert (key in keys(data)) "Expected keys not present in data"
    end
    N_samples = size(data["observations"])[2]
    @assert size(data["rewards"]) == (N_samples,) || size(data["rewards"]) == (1, N_samples)
    @assert size(data["terminals"]) == (N_samples,) || size(data["terminals"]) == (1, N_samples)
end