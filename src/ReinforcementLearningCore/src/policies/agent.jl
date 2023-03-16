export Agent

using Base.Threads: @spawn

using Functors: @functor

const cache_attrs = (:S, :SA, :SAR, :SART)

mutable struct AgentCache{S,A,R}
    state::S
    action::A
    reward::R
    terminal::Bool
    status::Symbol

    function AgentCache(policy::AbstractPolicy, env::AbstractEnv)
        new{typeof(policy(env)), typeof(state(env)), typeof(reward(env))}(policy(env), state(env), reward(env), false, :empty)
    end

    function AgentCache()
        new{Any, Any, Any}(0, 0, 0, false, :empty)
    end
end

sart_to_tuple(agent_cache::AgentCache{S,A,R}) where {S,A,R} = @NamedTuple{state::S, action::A, reward::R, terminal::Bool}((agent_cache.state::S, agent_cache.action::A, agent_cache.reward::R, agent_cache.terminal::Bool))

sar_to_tuple(agent_cache::AgentCache{S,A,R}) where {S,A,R} = @NamedTuple{state::S, action::A, reward::R}((agent_cache.state::S, agent_cache.action::A, agent_cache.reward::R))

sa_to_tuple(agent_cache::AgentCache{S,A,R}) where {S,A,R} = @NamedTuple{state::S, action::A}((agent_cache.state::S, agent_cache.action::A))

state_to_tuple(agent_cache::AgentCache{S,A,R}) where {S,A,R} = @NamedTuple{state::S}((agent_cache.state::S))

function RLBase.reset!(agent_cache::AgentCache)
    agent_cache.status = :empty
    return nothing
end

Base.isempty(agent_cache::AgentCache) = agent_cache.status == :empty

function update_state!(agent_cache::AgentCache, env::E) where {E <: AbstractEnv}
    agent_cache.state = state(env)
end

function update_reward!(agent_cache::AgentCache, env::E) where {E <: AbstractEnv}
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

function update_trajectory!(trajectory::Trajectory, agent_cache::AgentCache)
    if agent_cache.status == :sart
        push!(trajectory, sart_to_tuple(agent_cache))
    elseif agent_cache.status == :sar
        push!(trajectory, sar_to_tuple(agent_cache))
    elseif agent_cache.status == :sa
        push!(trajectory, sa_to_tuple(agent_cache))
    elseif agent_cache.status == :s
        push!(trajectory, state_to_tuple(agent_cache))
    end

    return
end

@functor Agent (policy,)

# !!! TODO: In async scenarios, parameters of the policy may still be updating
# (partially), which will result to incorrect action. This should be addressed
# in Oolong.jl with a wrapper
function (agent::Agent)(env::AbstractEnv, args...; kwargs...)
    action = agent.policy(env, args...; kwargs...)
    agent.cache.action = action
    
    if agent.cache.status == :s
        agent.cache.status = :sa
    end

    update_trajectory!(agent.trajectory, agent.cache)
    reset!(agent.cache)
    action
end

function (agent::Agent)(::PreActStage, env::AbstractEnv)
    update_state!(agent.cache, env)
    if agent.cache.status == :empty
        agent.cache.status = :s
    end
end

function (agent::Agent)(::PostActStage, env::E) where {E <: AbstractEnv}
    update_reward!(agent.cache, env)
    update_state!(agent.cache, env)
    agent.cache.terminal = is_terminated(env)
    agent.cache.status = :sart
    return
end
