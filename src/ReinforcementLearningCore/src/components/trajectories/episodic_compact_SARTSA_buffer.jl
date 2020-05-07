export EpisodicCompactSARTSATrajectory

using MacroTools: @forward

"""
    EpisodicCompactSARTSATrajectory(; state_type = Int, action_type = Int, reward_type = Float32, terminal_type = Bool)

Exactly the same with [`VectorialCompactSARTSATrajectory`](@ref). It only exists for multiple dispatch purpose.

!!! warning
    The `EpisodicCompactSARTSATrajectory` will not be automatically emptified when reaching the end of an episode.
"""
struct EpisodicCompactSARTSATrajectory{types,trace_types} <:
       AbstractTrajectory{SARTSA,types}
    trajectories::VectorialCompactSARTSATrajectory{types,trace_types}
end

EpisodicCompactSARTSATrajectory(; kwargs...) =
    EpisodicCompactSARTSATrajectory(VectorialCompactSARTSATrajectory(; kwargs...))

@forward EpisodicCompactSARTSATrajectory.trajectories Base.length,
Base.isempty,
Base.empty!,
Base.push!,
Base.pop!

# avoid method ambiguous
get_trace(t::EpisodicCompactSARTSATrajectory, s::Symbol) =
    get_trace(t.trajectories, s)
Base.getindex(t::EpisodicCompactSARTSATrajectory, i::Int) = getindex(t.trajectories, i)
Base.pop!(t::EpisodicCompactSARTSATrajectory, s::Symbol...) = pop!(t.trajectories, s...)
