export EpisodeTurnBuffer, episode_RTSA_buffer

"""
    EpisodeTurnBuffer{names, types, Tbs} <: AbstractTurnBuffer{names, types}
    EpisodeTurnBuffer{names, types}() where {names, types}

Similar to [`CircularTurnBuffer`](@ref), but instead of using [`CircularArrayBuffer`](@ref), it uses a vector to store each element specified by `names` and `types`. And when it reaches the end of an episode, the buffer is emptied first when a new observation is pushed.

!!! note
    Notice that, before emptifying the `EpisodeTurnBuffer`, the last element of each field is exracted and then pushed at the head of the buffer. Without this step, the first transition of the new episode will be lost!

See also: [`episode_RTSA_buffer`](@ref)
"""
struct EpisodeTurnBuffer{names,types,Tbs} <: AbstractTurnBuffer{names,types}
    buffers::Tbs
    function EpisodeTurnBuffer(configs::Pair{Symbol,DataType}...)
        buffers = merge(NamedTuple(), (name, Vector{type}()) for (name, type) in configs)
        names = Tuple(name for (name, _) in configs)
        types = Tuple{(type for (_, type) in configs)...}
        new{names,types,typeof(buffers)}(buffers)
    end
end

function Base.similar(b::EpisodeTurnBuffer{names,types}) where {names,types}
    EpisodeTurnBuffer(Pair.(names, types)...)
end

"""
    episode_RTSA_buffer(;kwargs...) -> EpisodeTurnBuffer

Initialize an `EpisodeTurnBuffer` with fields of **R**eward, **T**erminal, **S**tate, **A**ction.

# Keywords

- `state_eltype::Type=Int`: the type of state.
- `action_eltype::Type=Int`: the type of action.
- `reward_eltype::Type=Float32`: the type of reward.
- `terminal_eltype::Type=Bool`: the type of terminal.
"""
function episode_RTSA_buffer(
    ;
    state_eltype = Int,
    action_eltype = Int,
    reward_eltype = Float32,
    terminal_eltype = Bool,
)
    EpisodeTurnBuffer(
        :reward => reward_eltype,
        :terminal => terminal_eltype,
        :state => state_eltype,
        :action => action_eltype,
    )
end

"if the last turn is the terminal, then the EpisodeTurnBuffer is full"
isfull(b::EpisodeTurnBuffer{RTSA}) = length(b) > 0 && convert(Bool, b.buffers.terminal[end])

function Base.push!(b::EpisodeTurnBuffer{F}; kw...) where {F}
    if isfull(b)
        for f in F
            bf = getfield(buffers(b), f)
            x = bf[end]
            empty!(bf)
            push!(bf, x)
        end
    end
    for (k, v) in kw
        hasproperty(buffers(b), k) && push!(getproperty(buffers(b), k), v)
    end
end