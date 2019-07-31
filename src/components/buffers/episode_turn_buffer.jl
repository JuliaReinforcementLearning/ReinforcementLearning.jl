export EpisodeTurnBuffer, episode_SART_buffer

"""
    EpisodeTurnBuffer{names, types, Tbs} <: AbstractTurnBuffer{names, types}
    EpisodeTurnBuffer{names, types}() where {names, types}

Using a Vector to store each element specified by `names` and `types`.

See also: [`EpisodeSARTBuffer`](@ref), [`EpisodeSARTSBuffer`](@ref), [`EpisodeSARTSABuffer`](@ref)
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

capacity(b::EpisodeTurnBuffer) = isfull(b) ? length(b) : typemax(Int)

function episode_SART_buffer(
    ;state_eltype=Int,
    action_eltype=Int,
    reward_eltype=Float64,
    terminal_eltype=Bool)
    EpisodeTurnBuffer(
        :state => state_eltype,
        :action => action_eltype,
        :reward => reward_eltype,
        :terminal => terminal_eltype
    )
end

Base.length(b::EpisodeTurnBuffer{SART}) = length(b.buffers.terminal)

"if the last turn is the terminal, then the EpisodeTurnBuffer is full"
isfull(b::EpisodeTurnBuffer{SART}) = length(b) > 0 && convert(Bool, b.buffers.terminal[end])