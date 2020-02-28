export VectorialCompactSARTSATrajectory

const VectorialCompactSARTSATrajectory = Trajectory{
    SARTSA,
    types,
    NamedTuple{RTSA,trace_types},
} where {types,trace_types<:Tuple{Vararg{<:Vector}}}

"""
    VectorialCompactSARTSATrajectory(; state_type = Int, action_type = Int, reward_type = Float32, terminal_type = Bool)

This function creates a [`VectorialTrajectory`](@ref) of [`RTSA`](@ref) fields. Here the **Compact** in the function name means that, `state` and `next_state`, `action` and `next_action` reuse a same vector underlying.

# Example

```julia-repl
julia> t = VectorialCompactSARTSATrajectory()
0-element Trajectory{(:state, :action, :reward, :terminal, :next_state, :next_action),Tuple{Int64,Int64,Float32,Bool,Int64,Int64},NamedTuple{(:reward, :terminal, :state, :action),Tuple{Array{Float32,1},Array{Bool,1},Array{Int64,1},Array{Int64,1}}}}

julia> push!(t, state=0, action=0)

julia> push!(t, reward=0.f0, terminal=false, state=1, action=1)

julia> t
1-element Trajectory{(:state, :action, :reward, :terminal, :next_state, :next_action),Tuple{Int64,Int64,Float32,Bool,Int64,Int64},NamedTuple{(:reward, :terminal, :state, :action),Tuple{Array{Float32,1},Array{Bool,1},Array{Int64,1},Array{Int64,1}}}}:
 (state = 0, action = 0, reward = 0.0, terminal = 0, next_state = 1, next_action = 1)

julia> push!(t, reward=1.f0, terminal=true, state=2, action=2)

julia> t
2-element Trajectory{(:state, :action, :reward, :terminal, :next_state, :next_action),Tuple{Int64,Int64,Float32,Bool,Int64,Int64},NamedTuple{(:reward, :terminal, :state, :action),Tuple{Array{Float32,1},Array{Bool,1},Array{Int64,1},Array{Int64,1}}}}:
 (state = 0, action = 0, reward = 0.0, terminal = 0, next_state = 1, next_action = 1)
 (state = 1, action = 1, reward = 1.0, terminal = 1, next_state = 2, next_action = 2)

julia> get_trace(t, :state, :action)
(state = [0, 1], action = [0, 1])

julia> get_trace(t, :next_state, :next_action)
(next_state = [1, 2], next_action = [1, 2])

julia> pop!(t)
1-element Trajectory{(:state, :action, :reward, :terminal, :next_state, :next_action),Tuple{Int64,Int64,Float32,Bool,Int64,Int64},NamedTuple{(:reward, :terminal, :state, :action),Tuple{Array{Float32,1},Array{Bool,1},Array{Int64,1},Array{Int64,1}}}}:
 (state = 0, action = 0, reward = 0.0, terminal = 0, next_state = 1, next_action = 1)
```
"""
function VectorialCompactSARTSATrajectory(;
    state_type = Int,
    action_type = Int,
    reward_type = Float32,
    terminal_type = Bool,
)
    VectorialCompactSARTSATrajectory{
        Tuple{state_type,action_type,reward_type,terminal_type,state_type,action_type},
        Tuple{
            Vector{reward_type},
            Vector{terminal_type},
            Vector{state_type},
            Vector{action_type},
        },
    }((
        reward = Vector{reward_type}(),
        terminal = Vector{terminal_type}(),
        state = Vector{state_type}(),
        action = Vector{action_type}(),
    ))
end
