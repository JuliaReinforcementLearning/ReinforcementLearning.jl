export Agent

using Base.Threads: @spawn

using Functors: @functor

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
        agent = new{P,T, SRT}(policy, trajectory, SRT())

        if TrajectoryStyle(trajectory) === AsyncTrajectoryStyle()
            bind(trajectory, @spawn(optimise!(policy, trajectory)))
        end
        agent
    end

    function Agent(policy::P, trajectory::T, cache::C) where {P,T,C}
        agent = new{P,T,C}(policy, trajectory, cache)

        if TrajectoryStyle(trajectory) === AsyncTrajectoryStyle()
            bind(trajectory, @spawn(optimise!(policy, trajectory)))
        end
        agent
    end
end

Agent(;policy, trajectory, cache = SRT()) = Agent(policy, trajectory, cache)

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

function (agent::Agent)(::PreActStage, env::AbstractEnv)
    update!(agent, state(env))
end

(agent::Agent)(::PreActStage, player::Symbol, env::AbstractEnv) = (agent)(PreActStage(), env)

# !!! TODO: In async scenarios, parameters of the policy may still be updating
# (partially), which will result to incorrect action. This should be addressed
# in Oolong.jl with a wrapper
function (agent::Agent)(env::AbstractEnv, args...; kwargs...)
    action = agent.policy(env, args...; kwargs...)
    push!(agent.trajectory, agent.cache, action)
    action
end

function (agent::Agent)(::PostActStage, env::AbstractEnv)
    update!(agent.cache, reward(env), is_terminated(env))
end

function (agent::Agent)(::PostActStage, p::Symbol, env::AbstractEnv)
    update!(agent.cache, reward(env, p), is_terminated(env))
end

function (agent::Agent)(::PostExperimentStage, env::AbstractEnv)
    RLBase.reset!(agent.cache)
end

(agent::Agent)(::PostExperimentStage, p::Symbol, env::AbstractEnv) = (agent)(PostExperimentStage(), env)

function update!(agent::Agent, state::S) where {S}
    update!(agent.cache, state)
end

