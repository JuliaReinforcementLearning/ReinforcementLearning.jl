export CircularTurnBuffer, circular_RTSA_buffer, capacity, isfull, circular_PRTSA_buffer

import StatsBase: sample
using .Utils: SumTree

struct CircularTurnBuffer{names,types,Tbs} <: AbstractTurnBuffer{names,types}
    buffers::Tbs
end

function sample(b::CircularTurnBuffer; batch_size = 32, n_step = 1)
    inds = sample_indices(b, batch_size, n_step)
    inds, consecutive_view(b, inds, n_step)
end


#####
# RTSA
#####

function circular_RTSA_buffer(
    ;
    capacity,
    state_eltype = Int,
    state_size = (),
    action_eltype = Int,
    action_size = (),
    reward_eltype = Float64,
    reward_size = (),
    terminal_eltype = Bool,
    terminal_size = (),
)
    capacity += 1  # we need to store extra dummy (reward, terminal)
    buffers = (
        reward = CircularArrayBuffer{reward_eltype}(capacity, reward_size...),
        terminal = CircularArrayBuffer{terminal_eltype}(capacity, terminal_size...),
        state = CircularArrayBuffer{state_eltype}(capacity, state_size...),
        action = CircularArrayBuffer{action_eltype}(capacity, action_size...),
    )
    CircularTurnBuffer{
        RTSA,
        Tuple{reward_eltype,terminal_eltype,state_eltype,action_eltype},
        typeof(buffers),
    }(buffers)
end

sample_indices(b::CircularTurnBuffer{RTSA}, batch_size::Int, n_step::Int) =
    rand(1:length(b)-n_step, batch_size)

#####
# PRTSA
#####

function circular_PRTSA_buffer(
    ;
    capacity,
    state_eltype = Int,
    state_size = (),
    action_eltype = Int,
    action_size = (),
    reward_eltype = Float64,
    reward_size = (),
    terminal_eltype = Bool,
    terminal_size = (),
    priority_eltype = Float64,
)
    capacity += 1  # we need to store extra dummy (reward, terminal)
    buffers = (
        priority = SumTree(priority_eltype, capacity),
        reward = CircularArrayBuffer{reward_eltype}(capacity, reward_size...),
        terminal = CircularArrayBuffer{terminal_eltype}(capacity, terminal_size...),
        state = CircularArrayBuffer{state_eltype}(capacity, state_size...),
        action = CircularArrayBuffer{action_eltype}(capacity, action_size...),
    )
    CircularTurnBuffer{
        PRTSA,
        Tuple{priority_eltype,reward_eltype,terminal_eltype,state_eltype,action_eltype},
        typeof(buffers),
    }(buffers)
end

function sample_indices(b::CircularTurnBuffer{PRTSA}, batch_size::Int, n_step::Int)
    inds = Vector{Int}(undef, batch_size)
    for i = 1:length(inds)
        ind, p = sample(priority(b))
        while ind <= 1 || ind > length(b) - n_step + 1
            ind, p = sample(priority(b))
        end
        inds[i] = ind - 1  # !!! left shift by 1 because we are padding for priority
    end
    inds
end