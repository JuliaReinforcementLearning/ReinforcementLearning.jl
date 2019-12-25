export Trajectory

struct Trajectory{names,types,Tbs} <: AbstractTrajectory{names,types}
    buffers::Tbs
end

"A helper function to access inner fields"
Base.getindex(b::Trajectory, s::Symbol) = getproperty(b.buffers, s)

RLBase.get_trace(b::Trajectory, s::Symbol) = b[s]