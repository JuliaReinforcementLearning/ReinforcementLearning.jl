export CircularTurnBuffer, circular_RTSA_buffer, capacity, isfull

import StatsBase:sample

struct CircularTurnBuffer{names, types, Tbs} <: AbstractTurnBuffer{names, types}
    buffers::Tbs
    function CircularTurnBuffer(configs::Vararg{Pair{Symbol, NamedTuple{(:eltype, :capacity, :size), Tuple{DataType, Int, NTuple{M, Int}}}} where M})
        buffers = merge(NamedTuple(),
                        (n, CircularArrayBuffer{c.eltype}(c.capacity, c.size...)) for (n, c) in configs)
        names = Tuple(name for (name, _) in configs)
        types = Tuple{(config.eltype for (_, config) in configs)...}
        new{names, types, typeof(buffers)}(buffers)
    end
end

function circular_RTSA_buffer(
    ;capacity,
    state_eltype=Int,
    state_size=(),
    action_eltype=Int,
    action_size=(),
    reward_eltype = Float64,
    reward_size=(),
    terminal_eltype=Bool,
    terminal_size=()
)
    capacity += 1  # we need to store extra dummy (reward, terminal)
    CircularTurnBuffer(
        :reward => (eltype=reward_eltype, capacity=capacity, size=reward_size),
        :terminal => (eltype=terminal_eltype, capacity=capacity, size=terminal_size),
        :state => (eltype=state_eltype, capacity=capacity, size=state_size),
        :action => (eltype=action_eltype, capacity=capacity, size=action_size)
    )
end

sample_indices(b::CircularTurnBuffer{RTSA}, batch_size::Int, n_step::Int) = rand(1:length(b)-n_step, batch_size)

function sample(b::CircularTurnBuffer{RTSA}; batch_size=32, n_step=1)
    inds = sample_indices(b, batch_size, n_step)
    consecutive_view(b, inds, n_step)
end