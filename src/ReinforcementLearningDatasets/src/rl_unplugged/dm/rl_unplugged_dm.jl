export rl_unplugged_dm_dataset

using TFRecord

function make_batch_array(type::Type, feature_dims::Int, size::Tuple, batch_size::Int)
    Array{type, feature_dims+1}(undef, size..., batch_size)
end

function dm_buffer_dict(feature_size::Dict{String, Tuple}, batch_size::Int)
    obs_buffer = Dict{Symbol, AbstractArray}()

    buffer_dict = Dict{Symbol, Any}()

    for feature in keys(feature_size)
        feature_dims = length(feature_size[feature])
        if split(feature, "/")[1] == "observation"
            ob_key = Symbol(chop(feature, head=length("observation")+1, tail=0))
            if split(feature, "/")[end] == "egocentric_camera"
                obs_buffer[ob_key] = make_batch_array(UInt8, feature_dims, feature_size[feature], batch_size)
            else
                obs_buffer[ob_key] = make_batch_array(Float32, feature_dims, feature_size[feature], batch_size)
            end
        elseif feature == "action"
            buffer_dict[:action] = make_batch_array(Float32, feature_dims, feature_size[feature], batch_size)
            buffer_dict[:next_action] = make_batch_array(Float32, feature_dims, feature_size[feature], batch_size)
        elseif feature == "step_type"
            buffer_dict[:terminal] = make_batch_array(Bool, feature_dims, feature_size[feature], batch_size)
        else
            ob_key = Symbol(feature)
            buffer_dict[ob_key] = make_batch_array(Float32, feature_dims, feature_size[feature], batch_size)
        end
    end

    buffer_dict[:state] = deepcopy(NamedTuple(obs_buffer))
    buffer_dict[:next_state] = deepcopy(NamedTuple(obs_buffer))

    buffer_dict
end

function size_dict(game::String, type::String)
    if type == "dm_locomotion_humanoid"
        size_dict = DM_LOCOMOTION_HUMANOID_SIZE
    elseif type == "dm_locomotion_rodent"
        size_dict = DM_LOCOMOTION_RODENT_SIZE
    elseif type == "dm_control_suite"
        size_dict = DM_CONTROL_SUITE_SIZE[game]
    else
        error("given game type does not exist")
    end
end

function batch_named_tuple!(dest::NamedTuple, src::NamedTuple, i::Int)
    for fn in fieldnames(typeof(dest))
        xs = getfield(dest, fn)
        x = getfield(src, fn)
        if typeof(xs) <: NamedTuple
            batch_named_tuple!(xs, x, i)
        else
            selectdim(xs, ndims(xs), i) .= x
        end
    end
end

function make_transition(example::TFRecord.Example, feature_size::Dict{String, Tuple})
    f = example.features.feature
    
    observation_dict = Dict{Symbol, AbstractArray}()
    next_observation_dict = Dict{Symbol, AbstractArray}()
    transition_dict = Dict{Symbol, Any}()

    for feature in keys(feature_size)
        if split(feature, "/")[1] == "observation"
            ob_key = Symbol(chop(feature, head = length("observation")+1, tail=0))
            if split(feature, "/")[end] == "egocentric_camera"
                cam_feature_size = feature_size[feature]
                ob_size = prod(cam_feature_size)
                observation_dict[ob_key] = reshape(f[feature].bytes_list.value[1][1:ob_size], cam_feature_size...)
                next_observation_dict[ob_key] = reshape(f[feature].bytes_list.value[1][ob_size+1:end], cam_feature_size...)
            else
                if feature_size[feature] == ()
                    observation_dict[ob_key] = f[feature].float_list.value
                else
                    ob_size = feature_size[feature][1]
                    observation_dict[ob_key] = f[feature].float_list.value[1:ob_size]
                    next_observation_dict[ob_key] = f[feature].float_list.value[ob_size+1:end]
                end
            end
        elseif feature == "action"
            ob_size = feature_size[feature][1]
            action = f[feature].float_list.value
            transition_dict[:action] = action[1:ob_size]
            transition_dict[:next_action] = action[ob_size+1:end]
        elseif feature == "step_type"
            transition_dict[:terminal] = f[feature].float_list.value[1] == 2
        else
            ob_key = Symbol(feature)
            transition_dict[ob_key] = f[feature].float_list.value[1]
        end
    end
    state_nt = (state = NamedTuple(observation_dict),)
    next_state_nt = (next_state = NamedTuple(next_observation_dict),)
    transition = NamedTuple(transition_dict)

    merge(transition, state_nt, next_state_nt)
end
"""
    rl_unplugged_dm_dataset(game, shards; <keyword arguments>)

Returns a `RingBuffer`(@ref) of `NamedTuple` containing SARTS batches which supports 
multi threaded loading. Also contains additional data. The data enclosed within `:state` and
`next_state` is a NamedTuple consisting of all observations that are provided.
Check out keys in `DM_LOCOMOTION_HUMANOID`, `DM_LOCOMOTION_RODENT`, `DM_CONTROL_SUITE_SIZE` for supported 
datasets. Also check out `dm_params()` for more info on arguments.

# Arguments

- `game::String`: name of the dataset.
- `shards::Vector{Int}`: the shards that are to be loaded.
- `type::String`: type of the dm_env. can be `dm_control_suite`, `dm_locomotion_humanoid`, `dm_locomotion_rodent`.
- `is_shuffle::Bool`
- `stochasticity::Float32`: represents the stochasticity of the dataset. can be 
in the range: `0.0:0.1:0.5`. 
- `shuffle_buffer_size::Int=10_000`: size of the shuffle_buffer used in loading AtariRLTransitions.
- `tf_reader_bufsize::Int=10_000`: the size of the buffer `bufsize` that is used internally 
in `TFRecord.read`.
- `tf_reader_sz::Int=10_000`: the size of the `Channel`, `channel_size` that is returned by 
`TFRecord.read`.
- `batch_size::Int=256`: The number of samples within the batches that are returned by the `Channel`.
- `n_preallocations::Int=nthreads()*12`: the size of the buffer in the `Channel` that is returned.

!!! note

    To enable reading records from multiple files concurrently, remember to set the number of 
    threads correctly (See [JULIA_NUM_THREADS](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_NUM_THREADS)).
"""
function rl_unplugged_dm_dataset(
    game,
    shards;
    type = "dm_control_suite",
    is_shuffle = true,
    shuffle_buffer_size=10_000,
    tf_reader_bufsize=10_000,
    tf_reader_sz=10_000,
    batch_size=256,
    n_preallocations=nthreads()*12
)   
    n = nthreads()

    repo = "rl-unplugged-dm"
    
    folders= [
        @datadep_str "$repo-$game-$shard" 
        for shard in shards
    ]
    
    ch_files = Channel{String}(length(folders)) do ch
        for folder in cycle(folders)
            file = folder * "/$(readdir(folder)[1])"
            put!(ch, file)
        end
    end
    
    if is_shuffle
        files = buffered_shuffle(ch_files, length(folders))
    else
        files = ch_files
    end

    feature_size = size_dict(game, type)

    ch_src = Channel{NamedTuple}(n * tf_reader_sz) do ch
        for fs in partition(files, n)
            Threads.foreach(
                TFRecord.read(
                    fs;
                    compression=:gzip,
                    bufsize=tf_reader_bufsize,
                    channel_size=tf_reader_sz,
                );
                schedule=Threads.StaticSchedule()
            ) do x
                put!(ch, make_transition(x, feature_size))
            end
        end
    end

    if is_shuffle
        transitions = buffered_shuffle(
        ch_src,
        shuffle_buffer_size
        )
    else
        transitions = ch_src
    end
    
    taskref = Ref{Task}()
    
    buffer_dict = dm_buffer_dict(feature_size, batch_size)

    buffer = NamedTuple(buffer_dict)

    res = RingBuffer(buffer;taskref=taskref, sz=n_preallocations) do buff
        Threads.@threads for i in 1:batch_size
            batch_named_tuple!(buff, take!(transitions), i)
        end
    end

    bind(ch_src, taskref[])
    bind(ch_files, taskref[])
    res
end