export Trajectory

"""
    Trajectory{names,types,Tbs}(trajectories::Tbs)

A container of different `trajectories`.
Usually you won't use it directly.
"""
struct Trajectory{names,types,Tbs} <: AbstractTrajectory{names,types}
    trajectories::Tbs
end

"A helper function to access inner fields"
Base.getindex(t::Trajectory, s::Symbol) = getproperty(t.trajectories, s)

RLBase.get_trace(t::Trajectory, s::Symbol) = t[s]

function Base.push!(t::Trajectory, kv::Pair{Symbol})
    k, v = kv
    push!(t[k], v)
end

Base.pop!(t::Trajectory, s::Symbol) = pop!(t[s])
