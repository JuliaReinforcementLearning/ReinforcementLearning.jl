export VectorialTrajectory

const VectorialTrajectory = Trajectory{
    names,
    types,
    NamedTuple{names,trace_types},
} where {names,types,trace_types<:Tuple{Vararg{<:Vector}}}

"""
    VectorialTrajectory(;trace_name=trace_type ...)

Use `Vector` to store the traces.

# Example

```julia-repl
julia> t = VectorialTrajectory(;a=Int, b=Symbol)
0-element Trajectory{(:a, :b),Tuple{Int64,Symbol},NamedTuple{(:a, :b),Tuple{Array{Int64,1},Array{Symbol,1}}}}

julia> push!(t, a=0, b=:x)

julia> push!(t, a=1, b=:y)

julia> t
2-element Trajectory{(:a, :b),Tuple{Int64,Symbol},NamedTuple{(:a, :b),Tuple{Array{Int64,1},Array{Symbol,1}}}}:
 (a = 0, b = :x)
 (a = 1, b = :y)

julia> get_trace(t, :b)
2-element Array{Symbol,1}:
 :x
 :y

julia> pop!(t)

julia> t
1-element Trajectory{(:a, :b),Tuple{Int64,Symbol},NamedTuple{(:a, :b),Tuple{Array{Int64,1},Array{Symbol,1}}}}:
 (a = 0, b = :x)
```
"""
function VectorialTrajectory(; kwargs...)
    names = keys(kwargs.data)
    types = values(kwargs.data)
    trajectories =
        merge(NamedTuple(), (name, Vector{type}()) for (name, type) in zip(names, types))
    VectorialTrajectory{names,Tuple{types...},typeof(values(trajectories))}(trajectories)
end
