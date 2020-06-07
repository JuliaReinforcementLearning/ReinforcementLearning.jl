export PPOTrajectory

using MacroTools

struct PPOTrajectory{T<:CircularCompactSARTSATrajectory,P,names,types} <:
       AbstractTrajectory{names,types}
    trajectory::T
    action_log_prob::P
end

function PPOTrajectory(;
    capacity,
    action_log_prob_size = (),
    action_log_prob_type = Float32,
    kw...,
)
    t = CircularCompactSARTSATrajectory(; capacity = capacity, kw...)
    p = CircularArrayBuffer{action_log_prob_type}(action_log_prob_size..., capacity + 1)
    names = typeof(t).parameters[1]
    types = typeof(t).parameters[2]
    PPOTrajectory{
        typeof(t),
        typeof(p),
        (
            :state,
            :action,
            :action_log_prob,
            :reward,
            :terminal,
            :next_state,
            :next_action,
            :next_action_log_prob,
        ),
        Tuple{
            types.parameters[1:2]...,
            frame_type(p),
            types.parameters[3:end]...,
            frame_type(p),
        },
    }(
        t,
        p,
    )
end

MacroTools.@forward PPOTrajectory.trajectory Base.length, Base.isempty, RLCore.isfull

function RLCore.get_trace(t::PPOTrajectory, s::Symbol)
    if s == :action_log_prob
        select_last_dim(
            t.action_log_prob,
            1:(nframes(t.action_log_prob) > 1 ? nframes(t.action_log_prob) - 1 :
               nframes(t.action_log_prob)),
        )
    elseif s == :next_action_log_prob
        select_last_dim(t.action_log_prob, 2:nframes(t.action_log_prob))
    else
        get_trace(t.trajectory, s)
    end
end

Base.getindex(t::PPOTrajectory, s::Symbol) =
    s == :action_log_prob ? t.action_log_prob : t.trajectory[s]

function Base.getindex(p::PPOTrajectory, i::Int)
    s, a, r, t, s′, a′ = p.trajectory[i]
    (
        state = s,
        action = a,
        action_log_prob = select_last_dim(p.action_log_prob, i),
        reward = r,
        terminal = t,
        next_state = s′,
        next_action = a′,
        next_action_log_prob = select_last_dim(p.action_log_prob, i + 1),
    )
end

function Base.empty!(b::PPOTrajectory)
    empty!(b.action_log_prob)
    empty!(b.trajectory)
end

function Base.push!(b::PPOTrajectory, kv::Pair{Symbol})
    k, v = kv
    if k == :action_log_prob || k == :next_action_log_prob
        push!(b.action_log_prob, v)
    else
        push!(b.trajectory, kv)
    end
end

function Base.pop!(t::PPOTrajectory, s::Symbol)
    if s == :action_log_prob || s == :next_action_log_prob
        pop!(t.action_log_prob)
    else
        pop!(t.trajectory, s)
    end
end

function Base.pop!(t::PPOTrajectory)
    (pop!(t.trajectory)..., action_log_prob = pop!(t.action_log_prob))
end
