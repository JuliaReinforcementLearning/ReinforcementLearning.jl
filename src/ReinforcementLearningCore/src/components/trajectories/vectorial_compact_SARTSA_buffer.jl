export VectorialCompactSARTSATrajectory

const VectorialCompactSARTSATrajectory = Trajectory{
    SARTSA,
    types,
    NamedTuple{RTSA,trace_types},
} where {types,trace_types<:Tuple{Vararg{<:Vector}}}

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
