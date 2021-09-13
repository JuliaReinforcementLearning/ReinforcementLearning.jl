using NPZ
using CodecZlib
using Random

"""
Represents an `Iterable` dataset with the following fields:

# Fields
- `dataset::Dict{Symbol, Any}`: representation of the dataset as a Dictionary with style as `style`.
- `epochs::Vector{Int}`: list of epochs loaded.
- `repo::String`: the repository from which the dataset is taken.
- `length::Int`: the length of the dataset.
- `batch_size::Int`: the size of the batches returned by `iterate`.
- `style::Tuple{Symbol}`: the style of the `Iterator` that is returned, check out: [`SARTS`](@ref), [`SART`](@ref) and [`SA`](@ref) for types supported out of the box.
- `rng<:AbstractRNG`.
- `meta::Dict`: the metadata provided along with the dataset.
- `is_shuffle::Bool`: determines if the batches returned by `iterate` are shuffled.
"""
struct AtariDataSet{T<:AbstractRNG} <:RLDataSet
    dataset::Dict{Symbol, Any}
    epochs::Vector{Int}
    repo::String
    length::Int
    batch_size::Int
    style::Tuple
    rng::T
    meta::Dict
    is_shuffle::Bool
end

const samples_per_epoch = Int(1e6)
const atari_frame_size = 84
const epochs_per_game = 50

"""
    dataset(dataset, index, epochs; <keyword arguments>)

Create a dataset enclosed in a [`AtariDataSet`](@ref) `Iterable` type. Contain other related metadata
for the `dataset` that is passed. The returned type is an infinite or a finite `Iterator` 
respectively depending upon whether is_shuffle is `true` or `false`. For more information regarding
the dataset, refer to [google-research/batch_rl](https://github.com/google-research/batch_rl). Check out `atari_params()` for more info on arguments.

# Arguments

- `dataset::String`: name of the datset.
- `index::Int`: analogous to `v` and different values correspond to different `seed`s that are used for data collection. can be between `[1:5]`.
- `epochs::Vector{Int}`: list of epochs to load. included epochs should be between `[0:50]`.
- `style::NTuple=SARTS`: the style of the `Iterator` that is returned. can be [`SARTS`](@ref), [`SART`](@ref) or [`SA`](@ref).
- `repo::String="atari-replay-datasets"`: name of the repository of the dataset.
- `rng::AbstractRNG=StableRNG(123)`.
- `is_shuffle::Bool=true`: determines if the dataset is shuffled or not.
- `batch_size::Int=256` batch_size that is yielded by the iterator.

!!! warning

    The dataset takes up significant amount of space in RAM. Therefore it is advised to
    load even one epoch with 20GB of RAM. We are looking for ways to use lazy data loading here
    and any contributions are welcome.
"""
function dataset(
    game::String,
    index::Int,
    epochs::Vector{Int};
    style::NTuple=SARTS,
    repo::String="atari-replay-datasets",
    rng::AbstractRNG=MersenneTwister(123), 
    is_shuffle::Bool=true, 
    batch_size::Int=256
)
    
    try 
        @datadep_str "$repo-$game-$index"
    catch e
        if isa(e, KeyError)
            throw("Invalid params, check out `atari_params()`")
        end
    end
        
    path = @datadep_str "$repo-$game-$index" 

    @assert length(readdir(path)) == 1
    folder_name = readdir(path)[1]
    
    folder_path = "$path/$folder_name"
    files = readdir(folder_path)
    file_prefixes = collect(Set(map(x->join(split(x,"_")[1:2], "_"), files)))
    fields = map(collect(file_prefixes)) do x
        if split(x, "_")[1] == "\$store\$"
            x = split(x, "_")[2]
        else
            x = x
        end
    end

    s_epochs = Set(epochs)
    
    dataset = Dict()

    for (prefix, field) in zip(file_prefixes, fields)
        for epoch in s_epochs
            @assert epoch <= epochs_per_game
            data = open("$folder_path/$(prefix)_ckpt.$epoch.gz") do file
                stream = GzipDecompressorStream(file)
                NPZ.npzreadarray(stream)
            end

            if field == "observation"
                data = permutedims(data, [3, 2, 1]) # not sure about the orientation of the frame
            end

            if haskey(dataset, field)
                if field == "observation"
                    dataset[field] = cat(dataset[field], data, dims=3)
                else
                    dataset[field] = cat(dataset[field], data, dims=1)
                end
            else
                dataset[field] = data
            end
        end
    end

    num_epochs = length(s_epochs)

    atari_verify(dataset, num_epochs) 

    N_samples = size(dataset["observation"])[3]
    
    final_dataset = Dict{Symbol, Any}()
    meta = Dict{String, Any}()

    for (key, d_key) in zip(["observation", "action", "reward", "terminal"], Symbol.(["state", "action", "reward", "terminal"]))
            final_dataset[d_key] = dataset[key]
    end
    
    for key in keys(dataset)
        if !(key in ["observation", "action", "reward", "terminal"])
            meta[key] = dataset[key]
        end
    end

    return AtariDataSet(final_dataset, epochs, repo, N_samples, batch_size, style, rng, meta, is_shuffle)

end

function iterate(ds::AtariDataSet, state = 0)
    rng = ds.rng
    batch_size = ds.batch_size
    length = ds.length
    is_shuffle = ds.is_shuffle
    style = ds.style

    if is_shuffle
        inds = rand(rng, 1:length-1, batch_size)
    else
        if (state+1) * batch_size <= length
            inds = state*batch_size+1:(state+1)*batch_size
        else
            return nothing
        end
        state += 1
    end

    batch = (state = view(ds.dataset[:state], :, :, inds),
    action = view(ds.dataset[:action], inds),
    reward = view(ds.dataset[:reward], inds),
    terminal = view(ds.dataset[:terminal], inds))

    if style == SARTS
        batch = merge(batch, (next_state = view(ds.dataset[:state], :, :, (1).+(inds)),))
    end
    
    return batch, state
end


take(ds::AtariDataSet, n::Integer) = take(ds.dataset, n)
length(ds::AtariDataSet) = ds.length
IteratorEltype(::Type{AtariDataSet}) = EltypeUnknown() # see if eltype can be known (not sure about carla and adroit)

function atari_verify(dataset::Dict, num_epochs::Int)
    @assert size(dataset["observation"]) == (atari_frame_size, atari_frame_size, num_epochs*samples_per_epoch)
    @assert size(dataset["action"]) == (num_epochs * samples_per_epoch,)
    @assert size(dataset["reward"]) == (num_epochs * samples_per_epoch,)
    @assert size(dataset["terminal"]) == (num_epochs * samples_per_epoch,)
end
