export EpisodicCompactSARTSATrajectory

using MacroTools: @forward

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
RLBase.get_trace(t::EpisodicCompactSARTSATrajectory, s::Symbol) =
    get_trace(t.trajectories, s)
Base.getindex(t::EpisodicCompactSARTSATrajectory, i::Int) = getindex(t.trajectories, i)
Base.pop!(t::EpisodicCompactSARTSATrajectory, s::Symbol...) = pop!(t.trajectories, s...)
