export rl_unplugged_bsuite_dataset

using TFRecord

struct BSuiteRLTransition <: RLTransition
    state
    action
    reward
    terminal
    next_state
end

function BSuiteRLTransition(example::TFRecord.Example, game::String)
    f = example.features.feature
    if game == "catch"
        s = reshape(f["observation"].float_list.value[1:end÷2], 10, 5)
        s′ = reshape(f["observation"].float_list.value[end÷2+1:end], 10, 5)
    else
        s = f["observation"].float_list.value[1:end÷2]
        s′ = f["observation"].float_list.value[end÷2+1:end]
    end
    a = f["action"].int64_list.value[1]
    r = f["reward"].float_list.value[]
    t = f["step_type"].int64_list.value[1] == 2
    BSuiteRLTransition(s, a, r, t, s′)
end
"""
    rl_unplugged_bsuite_dataset(game, shards, type; <keyword arguments>)

Return a `RingBuffer`(@ref) of [`BSuiteRLTransition`](@ref) batches which supports 
multi threaded loading. Check out `bsuite_params()` for more info on arguments.

# Arguments

- `game::String`: name of the dataset. available datasets: `cartpole`, `mountain_car` and  `catch`. 
- `shards::Vector{Int}`: the shards that are to be loaded.
- `type::String`: can be `full`, `full_train` and `full_valid`.
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
function rl_unplugged_bsuite_dataset(
    game::String,
    shards::Vector{Int},
    type::String;
    is_shuffle::Bool=true,
    stochasticity::Float64=0.0,
    shuffle_buffer_size::Int=10_000,
    tf_reader_bufsize::Int=10_000,
    tf_reader_sz::Int=10_000,
    batch_size::Int=256,
    n_preallocations::Int=nthreads()*12
)   
    n = nthreads()

    repo = "rl-unplugged-bsuite"
    
    folders= [
        @datadep_str "$repo-$game-$stochasticity-$shard-$type" 
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
    
    ch_src = Channel{BSuiteRLTransition}(n * tf_reader_sz) do ch
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
                put!(ch, BSuiteRLTransition(x, game))
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

    ob_size = game=="mountain_car" ? 3 : 6 

    if game == "catch"
        obs_template = Array{Float32, 3}(undef, 10, 5, batch_size)
    else
        obs_template = Array{Float32, 2}(undef, ob_size, batch_size)
    end

    buffer = BSuiteRLTransition(
        copy(obs_template),
        Array{Int, 1}(undef, batch_size),
        Array{Float32, 1}(undef, batch_size),
        Array{Bool, 1}(undef, batch_size),
        copy(obs_template),
    )

    res = RingBuffer(buffer;taskref=taskref, sz=n_preallocations) do buff
        Threads.@threads for i in 1:batch_size
            batch!(buff, take!(transitions), i)
        end
    end

    bind(ch_src, taskref[])
    bind(ch_files, taskref[])
    res
end