using Random
using StableRNGs
using HDF5

import Base: iterate

export dataset
export SARTS
export SART

const SARTS = (:state, :action, :reward, :terminals, :next_state)
const SART = (:state, :action, :reward, :terminals)

# not exporting D4RLDataSet. (no need outside) -> Maybe make a constructor for D4RLDataSet that will return a D4RLDataSet dataset
# write docstring for this.
struct D4RLDataSet{T<:AbstractRNG}
    dataset::Dict{Symbol, Any}
    size::Int
    rng::T
    batch_size::Int64
    style::Tuple
    meta::Dict
end

# haven't included all the functionality like is_sequential, will be implemented soon
# Maybe enable the users providing their own paths to datasets if they already have it
# add an optional env_arg for additional checking
"""
    dataset(dataset::String; style::Tuple, rng<:AbstractRNG, is_shuffle::Bool, max_iters::Int64, batch_size::Int64)

Creates a dataset of enclosed in a D4RLDataSet type and other related metadata for the `dataset` that is passed.
The dataset type is an iterable that fetches batches when used in a for loop for convenience during offline training.

`dataset`: Name of the D4RLDataSet dataset.
`style`: the style of the iterator and the Dict inside D4RLDataSet that is returned.
`rng`: StableRNG
`max_iters`: maximum number of iterations for the iterator.
`is_shuffle`: whether the dataset is shuffled or not. `true` by default.
`batch_size`: batch_size that is yielded by the iterator. Defaults to 256.

The returned type is an infinite iterator which can be called using `iterate` and will return batches as specified in the dataset.
"""
function dataset(dataset::String;
    style=SARTS, 
    rng = StableRNG(123), 
    is_shuffle = true, 
    batch_size=256
)
    
    try 
        @datadep_str "d4rl-"*dataset 
    catch 
        throw("The provided dataset is not available") 
    end
        
    path = @datadep_str "d4rl-"*dataset

    @assert length(readdir(path)) == 1
    file_name = readdir(path)[1]
    
    # try to incorporate multi threading in loading stuff
    data = h5open(path*"/"*file_name, "r") do file
        read(file)
    end

    #check if we could obtain more info so that we could verify if the dataset actually matches the requirement
    #do some sanity checks on the data
    verify(data)

    dataset = Dict{Symbol, Any}()
    meta = Dict{String, Any}()

    N_samples = size(data["terminals"])[1]
    
    for (key, d_key) in zip(["observations", "actions", "rewards", "terminals"], Symbol.(["state", "action", "reward", "terminal"]))
            dataset[d_key] = data[key]
    end
    
    if style == SARTS
        dataset[:next_state] = @view data["observations"][:, 2:N_samples]
        dataset[:next_state] = cat(dataset[:next_state], data["observations"][:, 1]; dims = 2)
    end
    
    # put the data in separate container based on requirements
    for key in keys(data)
        if !(key in ["observations", "actions", "rewards", "terminals"])
            meta[key] = data[key]
        end
    end
    
    if is_shuffle
        # shuffle (try to incorporate multi threading in this)
        inds = shuffle(rng, 1:N_samples)
        
        for key in keys(dataset)
            dims = size(dataset[key])
            if length(dims) != 1
                dataset[key] = @view dataset[key][:, inds]
            else
                dataset[key] = @view dataset[key][inds]
            end
        end
    end

    return D4RLDataSet(dataset, N_samples, rng, batch_size, style, meta)

end

# making D4RLDataSet iterable
# temporary workaround for maintaining reproducibility (using seeds)
function iterate(ds::D4RLDataSet)
    rng = ds.rng
    # check return for dataset that is not shuffled dataset
    inds = rand(rng, 1:ds.size, ds.batch_size)

    batch = (state = copy(ds.dataset[:state][:, inds]),
    action = copy(ds.dataset[:action][:, inds]),
    reward = copy(ds.dataset[:reward][inds]),
    terminal = copy(ds.dataset[:terminal][inds]))

    if ds.style == SARTS
        batch = merge(batch, (next_state = copy(ds.dataset[:next_state][:, inds]),))
    end
    
    return batch
end

function verify(data::Dict{String, Any})
    for key in ["observations", "actions", "rewards", "terminals"]
        @assert (key in keys(data)) "Expected keys not present in data"
    end
    #if possible perform some sanity check to check if the shape matches the requirements of the environment.
    N_samples = size(data["observations"])[2]
    @assert size(data["rewards"]) == (N_samples,)
    @assert size(data["terminals"]) == (N_samples,)
end