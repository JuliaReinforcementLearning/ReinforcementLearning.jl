export EpisodeTurnBuffer, episode_RTSA_buffer

"""
    EpisodeTurnBuffer{names, types, Tbs} <: AbstractTurnBuffer{names, types}
    EpisodeTurnBuffer{names, types}() where {names, types}

Using a Vector to store each element specified by `names` and `types`.

See also: [`EpisodeRTSABuffer`](@ref), [`EpisodeRTSASBuffer`](@ref), [`EpisodeRTSASABuffer`](@ref)
"""
struct EpisodeTurnBuffer{names, types, Tbs} <: AbstractTurnBuffer{names, types}
    buffers::Tbs
    function EpisodeTurnBuffer(configs::Pair{Symbol, DataType}...)
        buffers = merge(NamedTuple(),
                        (name, Vector{type}()) for (name, type) in configs)
        names = Tuple(name for (name, _) in configs)
        types = Tuple{(type for (_, type) in configs)...}
        new{names, types, typeof(buffers)}(buffers)
    end
end


function episode_RTSA_buffer(
    ;state_eltype=Int,
    action_eltype=Int,
    reward_eltype=Float64,
    terminal_eltype=Bool)
    EpisodeTurnBuffer(
        :reward => reward_eltype,
        :terminal => terminal_eltype,
        :state => state_eltype,
        :action => action_eltype
    )
end

"if the last turn is the terminal, then the EpisodeTurnBuffer is full"
isfull(b::EpisodeTurnBuffer{RTSA}) = length(b) > 0 && convert(Bool, b.buffers.terminal[end])