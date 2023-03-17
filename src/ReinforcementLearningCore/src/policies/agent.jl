export Agent

using Base.Threads: @spawn

using Functors: @functor

mutable struct SART{S,A,R}
    state::Union{S, Missing}
    action::Union{A, Missing}
    reward::Union{R, Missing}
    terminal::Union{Bool, Missing}

    function SART()
        new{Any, Any, Any}(missing, missing, missing, missing)
    end
end

sart_to_tuple(sart::SART{S,A,R}) where {S,A,R} = @NamedTuple{state::Union{S,Missing}, action::Union{A,Missing}, reward::Union{R,Missing}, terminal::Union{Bool,Missing}}((sart.state, sart.action, sart.reward, sart.terminal))

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

    function Agent(policy::P, trajectory::T) where {P,T}
        agent = new{P,T, typeof(SART())}(policy, trajectory, SART())

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
    push!(agent.trajectory, sart_to_tuple(agent.cache))
    reset!(agent.cache)
    action
end

function (agent::Agent)(::PreActStage, env::AbstractEnv)
    update_state!(agent.cache, env)
end

function (agent::Agent)(::PostActStage, env::E) where {E <: AbstractEnv}
    update_reward!(agent.cache, env)
    agent.cache.terminal = is_terminated(env)
    return
end
