export CircularTrajectory

const CircularTrajectory = Trajectory{
    names,
    types,
    NamedTuple{names,trace_types},
} where {names,types,trace_types<:Tuple{Vararg{<:CircularArrayBuffer}}}

"""
    CircularTrajectory(; capacity, trace_name=eltype=>size...)

Similar to [`VectorialTrajectory`](@ref), but we use the
[`CircularArrayBuffer`](@ref) to store the traces. The `capacity`
here is used to specify the maximum length of the trajectory.

# Example

```julia-repl
julia> t = CircularTrajectory(capacity=10, state=Float64=>(3,3), reward=Int=>tuple())
0-element Trajectory{(:state, :reward),Tuple{Float64,Int64},NamedTuple{(:state, :reward),Tuple{CircularArrayBuffer{Float64,3},CircularArrayBuffer{Int64,1}}}}

julia> push!(t,state=rand(3,3), reward=1)

julia> push!(t,state=rand(3,3), reward=2)

julia> get_trace(t, :reward)
2-element CircularArrayBuffer{Int64,1}:
 1
 2

julia> get_trace(t, :state)
3×3×2 CircularArrayBuffer{Float64,3}:
[:, :, 1] =
 0.699906  0.382396   0.927411
 0.269807  0.0581324  0.239609
 0.222304  0.514408   0.318905

[:, :, 2] =
 0.956228  0.992505  0.109743
 0.763497  0.381387  0.540566
 0.223081  0.834308  0.634759

julia> pop!(t)

julia> get_trace(t, :state)
3×3×1 CircularArrayBuffer{Float64,3}:
[:, :, 1] =
 0.699906  0.382396   0.927411
 0.269807  0.0581324  0.239609
 0.222304  0.514408   0.318905

```
"""
function CircularTrajectory(; capacity, kwargs...)
    names = keys(kwargs.data)
    types_and_sizes = values(kwargs.data)
    types = (t for (t, s) in types_and_sizes)
    sizes = (s for (t, s) in types_and_sizes)
    trajectories = merge(
        NamedTuple(),
        (name, CircularArrayBuffer{t}(s..., capacity))
        for (name, (t, s)) in zip(names, types_and_sizes)
    )
    CircularTrajectory{names,Tuple{types...},typeof(values(trajectories))}(trajectories)
end

Base.length(t::CircularTrajectory) = maximum(nframes(x) for x in get_trace(t))
