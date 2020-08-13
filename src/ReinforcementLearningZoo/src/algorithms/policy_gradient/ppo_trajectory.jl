export PPOTrajectory

using MacroTools

const PPOTrajectory = CombinedTrajectory{
    <:SharedTrajectory{
        <:CircularArrayBuffer,
        <:NamedTuple{(:action_log_prob, :next_action_log_prob, :full_action_log_prob)},
    },
    <:CircularCompactSARTSATrajectory,
}

function PPOTrajectory(;
    capacity,
    action_log_prob_size = (),
    action_log_prob_type = Float32,
    kw...,
)
    CombinedTrajectory(
        SharedTrajectory(
            CircularArrayBuffer{action_log_prob_type}(
                action_log_prob_size...,
                capacity + 1,
            ),
            :action_log_prob,
        ),
        CircularCompactSARTSATrajectory(; capacity = capacity, kw...),
    )
end

const PPOActionMaskTrajectory = CombinedTrajectory{
    <:SharedTrajectory{
        <:CircularArrayBuffer,
        <:NamedTuple{(:action_log_prob, :next_action_log_prob, :full_action_log_prob)},
    },
    <:CircularCompactSALRTSALTrajectory,
}

function PPOActionMaskTrajectory(;
    capacity,
    action_log_prob_size = (),
    action_log_prob_type = Float32,
    kw...,
)
    CombinedTrajectory(
        SharedTrajectory(
            CircularArrayBuffer{action_log_prob_type}(
                action_log_prob_size...,
                capacity + 1,
            ),
            :action_log_prob,
        ),
        CircularCompactSALRTSALTrajectory(; capacity = capacity, kw...),
    )
end
