const CompactSARTSATrajectory =
    Union{CircularCompactSARTSATrajectory,VectorialCompactSARTSATrajectory}

function RLBase.get_trace(b::CompactSARTSATrajectory, s::Symbol)
    if s == :state || s == :action
        select_last_dim(b[s], 1:(length(b[s]) > 1 ? length(b[s]) - 1 : length(b[s])))
    elseif s == :reward || s == :terminal
        b[s]
    elseif s == :next_state
        select_last_dim(b[:state], 2:length(b[:state]))
    elseif s == :next_action
        select_last_dim(b[:action], 2:length(b[:action]))
    else
        throw(ArgumentError("unknown trace name: $s"))
    end
end

Base.length(b::CompactSARTSATrajectory) = length(b[:terminal])
Base.isempty(b::CompactSARTSATrajectory) = all(isempty(b[s]) for s in RTSA)

function Base.getindex(b::CompactSARTSATrajectory, i::Int)
    (
        state = select_last_dim(b[:state], i),
        action = select_last_dim(b[:action], i),
        reward = select_last_dim(b[:reward], i),
        terminal = select_last_dim(b[:terminal], i),
        next_state = select_last_dim(b[:state], i + 1),
        next_action = select_last_dim(b[:action], i + 1),
    )
end

function Base.empty!(b::CompactSARTSATrajectory)
    for s in RTSA
        empty!(b[s])
    end
    b
end

function Base.push!(b::CompactSARTSATrajectory, kv::Pair{Symbol})
    k, v = kv
    if k == :state || k == :next_state
        push!(b[:state], v)
    elseif k == :action || k == :next_action
        push!(b[:action], v)
    elseif k == :reward || k == :terminal
        push!(b[k], v)
    else
        throw(ArgumentError("unknown trace name: $k"))
    end
    b
end

function Base.pop!(t::CompactSARTSATrajectory, traces::Symbol...)
    for s in traces
        if s == :state || s == :next_state
            pop!(t[:state])
        elseif s == :action || s == :next_action
            pop!(t[:action])
        elseif s == :reward || s == :terminal
            pop!(t[s])
        else
            throw(ArgumentError("unknown trace name: $s"))
        end
    end
    t
end

function Base.pop!(t::CompactSARTSATrajectory)
    if length(t) <= 0
        throw(ArgumentError("can not pop! from an empty trajectory"))
    else
        pop!(t, :state)
        pop!(t, :action)
        pop!(t, :reward)
        pop!(t, :terminal)
    end
    t
end
