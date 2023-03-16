export Agent

using Base.Threads: @spawn

using Functors: @functor

const cache_attrs = (:state, :action, :reward, :terminal)

mutable struct AgentCache{S,A,R}
    state::Union{Missing, S}
    action::Union{Missing, A}
    reward::Union{Missing, R}
    terminal::Union{Missing, Bool}

    function AgentCache(policy::AbstractPolicy, env::AbstractEnv)
        new{typeof(policy(env)), typeof(state(env)), typeof(reward(env))}(missing, missing, missing)
    end

    function AgentCache()
        new{Any, Any, Any}(missing, missing, missing, missing)
    end
end

function struct_to_trajectory_tuple(agent_cache::AgentCache{S,A,R}) where {S,A,R}
    if !ismissing(agent_cache.terminal)
        return @NamedTuple{state::S, action::A, reward::R, terminal::Bool}((agent_cache.state, agent_cache.action, agent_cache.reward, agent_cache.terminal))
    elseif !ismissing(agent_cache.reward)
        return @NamedTuple{state::S, action::A, reward::R}((agent_cache.state, agent_cache.action, agent_cache.reward))
    elseif !ismissing(agent_cache.action)
        return @NamedTuple{state::S, action::A}((agent_cache.state, agent_cache.action))
    else
        return @NamedTuple{state::S}((agent_cache.state))
    end
end

function reset!(agent_cache::AgentCache{S,A,R}) where {S,A,R}
    agent_cache.state = missing
    agent_cache.action = missing
    agent_cache.reward = missing
    agent_cache.terminal = missing
    return nothing
end

function Base.isempty(agent_cache::AgentCache{S,A,R}) where {S,A,R}
    ismissing(agent_cache.action) & ismissing(agent_cache.state) & ismissing(agent_cache.reward) & ismissing(agent_cache.terminal)
end

function update_state!(agent_cache::AgentCache, env::E) where {E <: AbstractEnv}
    agent_cache.state = state(env)
end

function update_reward!(agent_cache::AgentCache{S,A,R}, env::E) where {S,A,R, E <: AbstractEnv}
    agent_cache.reward = reward(env)
end

"""
    Agent(;policy, trajectory) <: AbstractPolicy

A wrapper of an `AbstractPolicy`. Generally speaking, it does nothing but to
update the trajectory and policy appropriately in different stages. Agent
is a Callable and its call method accepts varargs and keyword arguments to be
passed to the policy. 

"""
mutable struct Agent{P,T,C} <: AbstractPolicy
    policy::P
    trajectory::T
    cache::C # trajectory does not support partial inserting

    function Agent(policy::P, trajectory::T; env=missing) where {P,T}
        if !ismissing(env)
            cache = AgentCache(policy, env)
            agent = new{P,T, typeof(cache)}(policy, trajectory, cache)
        else
            agent = new{P,T, typeof(AgentCache())}(policy, trajectory, AgentCache())
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
    push!(agent.trajectory, struct_to_trajectory_tuple(agent.cache))
    reset!(agent.cache)
    action
end

(agent::Agent)(::PreActStage, env::AbstractEnv) = update_state!(agent.cache, env)

function (agent::Agent)(::PostActStage, env::E) where {E <: AbstractEnv}
    update_reward!(agent.cache, env)
    update_state!(agent.cache, env)
    agent.cache.terminal = is_terminated(env)
    return
end
