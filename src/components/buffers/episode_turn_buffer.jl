export EpisodeTurnBuffer

"""
    EpisodeTurnBuffer{names, types, Tbs} <: AbstractTurnBuffer{names, types}
    EpisodeTurnBuffer{names, types}() where {names, types}

Using a Vector to store each element specified by `names` and `types`.

See also: [`EpisodeSARDBuffer`](@ref), [`EpisodeSARDSBuffer`](@ref), [`EpisodeSARDSABuffer`](@ref)
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

##############################
# EpisodeSARDBuffer
##############################
function EpisodeSARDBuffer(
    ;state_type,
    action_type=Int,
    reward_type=Float64,
    isdone_type=Bool)
    EpisodeTurnBuffer(
        :state => state_type,
        :action => action_type,
        :reward => reward_type,
        :isdone => isdone_type
    )
end