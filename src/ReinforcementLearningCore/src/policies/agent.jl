export Agent

using Base.Threads: @spawn

using Functors: @functor

struct SRT{S,R,T}
    state::S
    reward::R
    terminal::T

    function SRT()
        new{Nothing, Nothing, Nothing}(nothing, nothing, nothing, nothing)
    end

    function SRT{S,R,T}(state::S, reward::R, terminal::T) where {S,R,T}
        new{S,A,R,T}(state, reward, terminal)
    end

    function SRT(sart::SRT{S,R,T}, state::S) where {S,R,T}
        new{S,R,T}(state, sart.reward, sart.terminal)
    end
end

function Base.push!(t::Trajectory, sart::SRT{S,Nothing,Nothing}, action::A) where {S,A}
    push!(t, @NamedTuple{state::S, action::A}((sart.state, action)))
end

function Base.push!(t::Trajectory, sart::SRT{S,R,T}, action::A) where {S,A,R,T}
    push!(t, @NamedTuple{state::S, action::A, reward::R, terminal::T}((sart.state, action, sart.reward, sart.terminal)))
end

Base.push!(cache::SRT{Nothing,R,T}, state::S) where {S,R,T} = agent.cache = SRT{S,R,T}(agent.cache, state)

Base.isempty(sart::SRT{Nothing,Nothing,Nothing}) = true
Base.isempty(sart::SRT) = false

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
    cache::C # need cache to collect elements as trajectory does not support partial inserting

    function Agent(policy::P, trajectory::T) where {P,T}
        agent = new{P,T, typeof(SRT())}(policy, trajectory, SRT())

        if TrajectoryStyle(trajectory) === AsyncTrajectoryStyle()
            bind(trajectory, @spawn(optimise!(policy, trajectory)))
        end
        agent
    end
end

Agent(; policy, trajectory) = Agent(policy, trajectory)

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
    push!(agent.trajectory, agent.cache, action)
    action
end

function (agent::Agent)(::PreActStage, env::AbstractEnv)
    agent.cache = push!(agent.cache, state(env))
end

function (agent::Agent)(::PostActStage, env::E) where {E <: AbstractEnv}
    agent.cache = SRT{Nothing, Nothing, Any, Bool}(nothing, reward(env), is_terminated(env))
    return
end
