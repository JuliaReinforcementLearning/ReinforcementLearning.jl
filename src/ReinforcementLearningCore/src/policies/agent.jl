export Agent

using Base.Threads: @spawn

using Functors: @functor

mutable struct SART{S,A,R}
    state::Union{S, Missing}
    action::Union{A, Missing}
    reward::Union{R, Missing}
    terminal::Union{Bool, Missing}

    function SART(policy::AbstractPolicy, env::AbstractEnv)
        new{typeof(state(env)), typeof(policy(env)), typeof(reward(env))}(missing, missing, missing, missing)
    end

    function SART()
        new{Any, Any, Any}(missing, missing, missing, missing)
    end
end

struct SART_strict{S,A,R}
    state::S
    action::A
    reward::R
    terminal::Bool

    function SART_strict(sart::SART{S,A,R}) where {S,A,R}
        new{S,A,R}(sart.state, sart.action, sart.reward, sart.terminal)
    end
end

struct StateAgent_strict{S,A}
    state::S
    action::A

    function StateAgent_strict(sart::SART{S,A,R}) where {S,A,R}
        new{S,A}(sart.state, sart.action)
    end
end

sart_to_tuple(sart::SART_strict{S,A,R}) where {S,A,R} = @NamedTuple{state::S, action::A, reward::R, terminal::Bool}((sart.state::S, sart.action::A, sart.reward::R, sart.terminal::Bool))

sart_to_tuple(sart::SART) = sart_to_tuple(SART_strict(sart))

state_agent_to_tuple(sa::StateAgent_strict{S,A}) where {S,A} = @NamedTuple{state::S, action::A}((sa.state::S, sa.action::A))

state_agent_to_tuple(sart::SART) = state_agent_to_tuple(StateAgent_strict(sart))

function RLBase.reset!(sart::SART)
    sart.state = missing
    sart.action = missing
    sart.reward = missing
    sart.terminal = missing
    return nothing
end

Base.isempty(sart::SART) = ismissing(sart.state) && ismissing(sart.action) && ismissing(sart.reward) && ismissing(sart.terminal)

function update_state!(sart::SART, env::E) where {E <: AbstractEnv}
    sart.state = state(env)
end

function update_reward!(sart::SART, env::E) where {E <: AbstractEnv}
    sart.reward = reward(env)
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
            cache = SART(policy, env)
            agent = new{P,T, typeof(cache)}(policy, trajectory, cache)
        else
            agent = new{P,T, typeof(SART())}(policy, trajectory, SART())
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
    if !ismissing(agent.cache.terminal)
        push!(agent.trajectory, sart_to_tuple(agent.cache))
    else
        push!(agent.trajectory, state_agent_to_tuple(agent.cache))
    end
    reset!(agent.cache)
    action
end

function (agent::Agent)(::PreActStage, env::AbstractEnv)
    update_state!(agent.cache, env)
end

function (agent::Agent)(::PostActStage, env::E) where {E <: AbstractEnv}
    update_reward!(agent.cache, env)
    update_state!(agent.cache, env)
    agent.cache.terminal = is_terminated(env)
    return
end
