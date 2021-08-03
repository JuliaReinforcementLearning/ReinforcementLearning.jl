"""
Represents an iterable dataset of type AtariDataSet with the following fields:

`dataset`: Dict{Symbol, Any}, representation of the dataset as a Dictionary with style as `style`
`index`: Int, represents an init seed
`epochs`: Vector{Int}, list of epochs to load
`repo`: String, the repository from which the dataset is taken
`size`: Integer, the size of the dataset
`batch_size`: Integer, the size of the batches returned by `iterate`.
`style`: Tuple, the type of the NamedTuple, for now SARTS and SART is supported.
`rng`<: AbstractRNG.
`meta`: Dict, the metadata provided along with the dataset
`is_shuffle`: Bool, determines if the batches returned by `iterate` are shuffled.
"""
struct AtariDataSet{T<:AbstractRNG} <:RLDataSet
    dataset::Dict{Symbol, Any}
    epochs::Vector{Int}
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
    atari_dataset(dataset::String, epochs::Vector{Int}; repo::String, style::Tuple, rng<:AbstractRNG, is_shuffle::Bool, max_iters::Int64, batch_size::Int64)

Creates a dataset of enclosed in a AtariDataSet type and other related metadata for the `dataset` that is passed.
The `AtariDataSet` type is an iterable that fetches batches when used in a for loop for convenience during offline training.

`dataset`: String, name of the datset.
`epochs`: Vector{Int}, list of epochs to load
`repo`: Name of the repository of the dataset
`style`: the style of the iterator and the Dict inside AtariDataSet that is returned.
`rng`: StableRNG
`max_iters`: maximum number of iterations for the iterator.
`is_shuffle`: whether the dataset is shuffled or not. `true` by default.
`batch_size`: batch_size that is yielded by the iterator. Defaults to 256.

The returned type is an infinite iterator which can be called using `iterate` and will return batches as specified in the dataset.
"""
function dataset(game::String,
    index::Int,
    epochs::Vector{Int};
    style=SARTS,
    repo = "atari-replay-datasets",
    rng = StableRNG(123), 
    is_shuffle = true, 
    batch_size=256
)
    
    try 
        @datadep_str "$repo-$game-$index"
    catch
        throw("The provided dataset is not available") 
    end
        
    path = @datadep_str "$repo-$game-$index" 

    @assert length(readdir(path)) == 1
    folder_name = readdir(path)[1]
    
    folder_path = "$path/$folder_name"
    files = readdir(home_path)
    file_prefixes = map(x->join(split(x,"_")[1:2], "_"), files)
    set_file_prefixes = collect(Set(file_prefixes))
    fields = map(collect(set_file_prefixes)) do x
        if split(x, "_")[1] == "\$store\$"
            x = split(x, "_")[2]
        else
            x = x
        end
    end

    s_epochs = Set(epochs)
    
    dataset = Dict{Symbol, Any}()

    for (prefix, field) in zip(set_file_prefixes, fields)
        for epoch in s_epochs
            data = gzopen("$folder_path/$(prefix)_ckpt.$epoch.gz") do file
                npzreadarray(file)
            end

            if field == "observation"
                data = permutedims(data, [3, 2, 1]) # not sure about the orientation of the frame
            end

            if haskey(dataset, field)
                if field == "observation"
                    dataset[field] = cat(dataset[field], data, dims=3)
                else
                    dataset[field] = cat(dataset[field], data, dims=2)
                end
            else
                dataset[field] = data
            end    
        end
    end

    num_epochs = length(s_epochs)

    atari_verify(dataset, num_epochs) 

    N_samples = size(data["observation"])[3]
    
    final_dataset = Dict{String, Any}()
    meta = Dict{String, Any}()

    for (key, d_key) in zip(["observation", "action", "reward", "terminal"], Symbol.(["state", "action", "reward", "terminal"]))
            final_dataset[d_key] = dataset[key]
    end
    
    for key in keys(data)
        if !(key in ["observation", "action", "reward", "terminal"])
            meta[key] = dataset[key]
        end
    end

    return AtariDataSet(final_dataset, epochs, repo, N_samples, batch_size, style, rng, meta, is_shuffle)

end

function iterate(ds::AtariDataSet, state = 0)
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

    batch = (state = copy(ds.dataset[:state][:, :, inds]),
    action = copy(ds.dataset[:action][inds]),
    reward = copy(ds.dataset[:reward][inds]),
    terminal = copy(ds.dataset[:terminal][inds]))

    if style == SARTS
        batch = merge(batch, (next_state = copy(ds.dataset[:state][:, (1).+(inds)]),))
    end
    
    return batch, state
end


take(ds::AtariDataSet, n::Integer) = take(ds.dataset, n)
length(ds::AtariDataSet) = ds.size
IteratorEltype(::Type{AtariDataSet}) = EltypeUnknown() # see if eltype can be known (not sure about carla and adroit)

function atari_verify(dataset::Dict{String, Any}, num_epochs::Int)
    @assert size(dataset["observation"]) == (84, 84, num_epochs*Int(1e6))
    @assert size(dataset["action"]) == (Int(1e6),)
    @assert size(dataset["reward"]) == (Int(1e6),)
    @assert size(dataset["terminal"]) == (Int(1e6),)
end