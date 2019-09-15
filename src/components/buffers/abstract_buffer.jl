export AbstractTurnBuffer, buffers, RTSA, isfull, consecutive_view

"""
    AbstractTurnBuffer{names, types} <: AbstractArray{NamedTuple{names, types}, 1}

`AbstractTurnBuffer` is supertype of a collection of buffers to store the interactions between agents and environments.
It is a subtype of `AbstractArray{NamedTuple{names, types}, 1}` where `names` specifies which fields are to store
and `types` is the coresponding types of the `names`.


| Required Methods| Brief Description |
|:----------------RTSA----------------|
| `Base.push!(b::AbstractTurnBuffer{names, types}, s[, a, r, d, s′, a′])` | Push a turn info into the buffer. According to different `names` and `types` of the buffer `b`, it may accept different number of arguments |
| `isfull(b)` | Check whether the buffer is full or not |
| `Base.length(b)` | Return the length of buffer |
| `Base.getindex(b::AbstractTurnBuffer{names, types})` | Return a turn of type `NamedTuple{names, types}` |
| `Base.empty!(b)` | Reset the buffer |
| **Optional Methods** | |
| `Base.size(b)` | Return `(length(b),)` by default |
| `Base.isempty(b)` | Check whether the buffer is empty or not. Return `length(b) == 0` by default |
| `Base.lastindex(b)` | Return `length(b)` by default |
"""
abstract type AbstractTurnBuffer{names,types} <: AbstractArray{NamedTuple{names,types},1} end

buffers(b::AbstractTurnBuffer) = getfield(b, :buffers)

Base.size(b::AbstractTurnBuffer) = (length(b),)
Base.lastindex(b::AbstractTurnBuffer) = length(b)
Base.isempty(b::AbstractTurnBuffer) = all(isempty(x) for x in buffers(b))
Base.empty!(b::AbstractTurnBuffer) =
    for x in buffers(b)
        empty!(x)
    end
Base.getindex(b::AbstractTurnBuffer{names,types}, i::Int) where {names,types} =
    NamedTuple{names,types}(Tuple(x[i] for x in buffers(b)))
isfull(b::AbstractTurnBuffer) = all(isfull(x) for x in buffers(b))

function Base.push!(b::AbstractTurnBuffer; kw...)
    for (k, v) in kw
        hasproperty(buffers(b), k) && push!(getproperty(buffers(b), k), v)
    end
end

function Base.push!(b::AbstractTurnBuffer, experience::Pair{<:Observation})
    obs, a = experience
    push!(
        b;
        state = get_state(obs),
        reward = get_reward(obs),
        terminal = get_terminal(obs),
        action = a,
        obs.meta...,
    )
end

#####
# RTSA (Reward, Terminals, State, Action)
#####

state(b::AbstractTurnBuffer) = buffers(b).state
action(b::AbstractTurnBuffer) = buffers(b).action
reward(b::AbstractTurnBuffer) = buffers(b).reward
terminal(b::AbstractTurnBuffer) = buffers(b).terminal

const RTSA = (:reward, :terminal, :state, :action)

Base.getindex(b::AbstractTurnBuffer{RTSA,types}, i::Int) where {types} =
    (
     state = state(b)[i],
     action = action(b)[i],
     reward = reward(b)[i+1],
     terminal = terminal(b)[i+1],
     next_state = state(b)[i+1],
     next_action = action(b)[i+1],
    )

#####
# PRTSA (Prioritized, Reward, Terminal, State, Action)
#####

const PRTSA = (:priority, :reward, :terminal, :state, :action)

priority(b::AbstractTurnBuffer) = buffers(b).priority

Base.getindex(b::AbstractTurnBuffer{PRTSA,types}, i::Int) where {types} =
    (
     state = state(b)[i],
     action = action(b)[i],
     reward = reward(b)[i+1],
     terminal = terminal(b)[i+1],
     next_state = state(b)[i+1],
     next_action = action(b)[i+1],
     priority = priority(b)[i+1],
    )

function consecutive_view(b::AbstractTurnBuffer, inds, n)
    next_inds = inds .+ 1

    (
     states = consecutive_view(state(b), inds, n),
     actions = consecutive_view(action(b), inds, n),
     rewards = consecutive_view(reward(b), next_inds, n),
     terminals = consecutive_view(terminal(b), next_inds, n),
     next_states = consecutive_view(state(b), next_inds, n),
     next_actions = consecutive_view(action(b), next_inds, n),
    )
end

function extract_SARTS(batch, γ)
    n_step, batch_size = size(batch.terminals)
    states = selectdim(batch.states, ndims(batch.states) - 1, 1)
    actions = selectdim(batch.actions, ndims(batch.actions) - 1, 1)
    next_states = selectdim(batch.next_states, ndims(batch.next_states) - 1, n_step)

    rewards, terminals = zeros(Float32, batch_size), fill(false, batch_size)

    # make sure that we only consider experiences in current episode
    for i = 1:batch_size
        t = findfirst(view(batch.terminals, :, i))

        if isnothing(t)
            terminals[i] = false
            rewards[i] = discount_rewards_reduced(view(batch.rewards[:, i]), γ)
        else
            terminals[i] = true
            rewards[i] = discount_rewards_reduced(view(batch.rewards[1:t, i]), γ)
        end
    end

    states, actions, rewards, terminals, next_states
end

Base.length(b::AbstractTurnBuffer) = max(0, length(terminal(b)) - 1)