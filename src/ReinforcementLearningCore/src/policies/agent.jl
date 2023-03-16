export Agent

using Base.Threads: @spawn

using Functors: @functor

mutable struct AgentCache{A,S,R}
    action::Union{Missing, A}
    state::Union{Missing, S}
    reward::Union{Missing, R}
    terminal::Union{Missing, Bool}

    function AgentCache(policy::AbstractPolicy, env::AbstractEnv)
        new{typeof(policy(env)), typeof(state(env)), typeof(reward(env))}(missing, missing, missing)
    end

    function AgentCache()
        new{Any, Any, Any}(missing, missing, missing, missing)
    end
end

const cache_attrs = (:action, :state, :reward, :terminal)

"""
    Agent(;policy, trajectory) <: AbstractPolicy

A wrapper of an `AbstractPolicy`. Generally speaking, it does nothing but to
update the trajectory and policy appropriately in different stages. Agent
is a Callable and its call method accepts varargs and keyword arguments to be
passed to the policy. 

"""
mutable struct Agent{P,T} <: AbstractPolicy
    policy::P
    trajectory::T
    cache::AgentCache # trajectory does not support partial inserting

    function Agent(policy::P, trajectory::T; env=missing) where {P,T}
        if !ismissing(env)
            agent = new{P,T}(policy, trajectory, AgentCache(policy, env))
        else
            agent = new{P,T}(policy, trajectory, AgentCache())
        end

        if TrajectoryStyle(trajectory) === AsyncTrajectoryStyle()
            bind(trajectory, @spawn(optimise!(policy, trajectory)))
        end
        agent
    end
end

Agent(; policy, trajectory, env=missing) = Agent(policy, trajectory, cache)

RLBase.optimise!(agent::Agent) = optimise!(TrajectoryStyle(agent.trajectory), agent)
RLBase.optimise!(::SyncTrajectoryStyle, agent::Agent) =
    optimise!(agent.policy, agent.trajectory)

# already spawn a task to optimise inner policy when initializing the agent
RLBase.optimise!(::AsyncTrajectoryStyle, agent::Agent) = nothing

function RLBase.optimise!(policy::AbstractPolicy, trajectory::Trajectory)
    for batch in trajectory
        optimise!(policy, batch)
    end
end

@functor Agent (policy,)

# !!! TODO: In async scenarios, parameters of the policy may still be updating
# (partially), which will result to incorrect action. This should be addressed
# in Oolong.jl with a wrapper
function (agent::Agent)(env::AbstractEnv, args...; kwargs...)
    action = agent.policy(env, args...; kwargs...)
    agent.cache.action = action
    push!(agent.trajectory, struct_to_namedtuple(agent.cache))
    reset!(agent.cache)
    action
end

function struct_to_namedtuple(agent_cache::AgentCache{S,R}) where {S,R}
    attr_is_not_missing = Bool[!ismissing(getfield(agent_cache, f)) for f in cache_attrs]
    present_attrs = cache_attrs[attr_is_not_missing]
    return NamedTuple{present_attrs}(getfield(agent_cache, f) for f in present_attrs)
end

function reset!(agent_cache::AgentCache{S,R}) where {S,R}
    agent_cache.action = missing
    agent_cache.state = missing
    agent_cache.reward = missing
    agent_cache.terminal = missing
    return nothing
end

function Base.isempty(agent_cache::AgentCache{S,R}) where {S,R}
    ismissing(agent_cache.action) & ismissing(agent_cache.state) & ismissing(agent_cache.reward) & ismissing(agent_cache.terminal)
end

(agent::Agent)(::PreActStage, env::E) where {E <: AbstractEnv} =
    agent.cache.state = state(env)
    return

function (agent::Agent)(::PostActStage, env::E) where {E <: AbstractEnv}
    agent.cache.reward = reward(env)
    agent.cache.state = state(env)
    agent.cache.terminal = is_terminated(env)
    return
end
