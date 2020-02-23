export CircularCompactSARTSATrajectory

const CircularCompactSARTSATrajectory = Trajectory{
    SARTSA,
    T1,
    NamedTuple{RTSA,T2},
} where {T1,T2<:Tuple{Vararg{<:CircularArrayBuffer}}}

function CircularCompactSARTSATrajectory(;
    capacity,
    state_type = Int,
    state_size = (),
    action_type = Int,
    action_size = (),
    reward_type = Float32,
    reward_size = (),
    terminal_type = Bool,
    terminal_size = (),
)
    capacity > 0 || throw(ArgumentError("capacity must > 0"))
    CircularCompactSARTSATrajectory{
        Tuple{state_type,action_type,reward_type,terminal_type,state_type,action_type},
        Tuple{
            CircularArrayBuffer{reward_type,length(reward_size) + 1},
            CircularArrayBuffer{terminal_type,length(terminal_size) + 1},
            CircularArrayBuffer{state_type,length(state_size) + 1},
            CircularArrayBuffer{action_type,length(action_size) + 1},
        },
    }((
        reward = CircularArrayBuffer{reward_type}(reward_size..., capacity),
        terminal = CircularArrayBuffer{terminal_type}(terminal_size..., capacity),
        state = CircularArrayBuffer{state_type}(state_size..., capacity + 1),
        action = CircularArrayBuffer{action_type}(action_size..., capacity + 1),
    ))
end
