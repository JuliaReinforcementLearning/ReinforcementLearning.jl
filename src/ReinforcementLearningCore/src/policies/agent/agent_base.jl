export Agent

using Base.Threads: @spawn

using Functors: @functor
import Base.push!
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

    function Agent(policy::P, trajectory::T) where {P<:AbstractPolicy, T<:Trajectory}
        agent = new{P,T}(policy, trajectory)

        if TrajectoryStyle(trajectory) === AsyncTrajectoryStyle()
            bind(trajectory, @spawn(optimise!(policy, trajectory)))
        end
        agent
    end
end

Agent(;policy, trajectory) = Agent(policy, trajectory)

RLBase.optimise!(agent::Agent, stage::S) where {S<:AbstractStage} = RLBase.optimise!(TrajectoryStyle(agent.trajectory), agent, stage)
RLBase.optimise!(::SyncTrajectoryStyle, agent::Agent, stage::S) where {S<:AbstractStage} = RLBase.optimise!(agent.policy, stage, agent.trajectory)

# already spawn a task to optimise inner policy when initializing the agent
RLBase.optimise!(::AsyncTrajectoryStyle, agent::Agent, stage::S) where {S<:AbstractStage} = nothing

#by default, optimise does nothing at all stage
function RLBase.optimise!(policy::AbstractPolicy, stage::AbstractStage, trajectory::Trajectory) end

@functor Agent (policy,)

function Base.push!(agent::Agent, ::PreEpisodeStage, env::AbstractEnv)
    push!(agent.trajectory, (state = state(env),))
end

# !!! TODO: In async scenarios, parameters of the policy may still be updating
# (partially), which will result to incorrect action. This should be addressed
# in Oolong.jl with a wrapper
@generated function RLBase.plan!(agent::Agent{<:Any, Trajectory{EpisodesBuffer{names}}}, env::AbstractEnv)
    if :action_log_prob in names
        return :(RLBase.plan!(agent.policy, env))
    else
        return :(RLBase.plan!(agent.policy, env, is_return_log_prob = true)) #returns a tuple with the action and the log_prob.
        #Note: it is expected that a policy that needs is_return_log_prob will implement a plan! function that accepts the above arguments.
    end

end

function Base.push!(agent::Agent, ::PostActStage, env::AbstractEnv, action)
    next_state = state(env)
    push!(agent.trajectory, (state = next_state, action = action, reward = reward(env), terminal = is_terminated(env)))
end

function Base.push!(agent::Agent, ::PostActStage, env::AbstractEnv, action::Tuple)
    next_state = state(env)
    push!(agent.trajectory, (state = next_state, action = action[1], action_log_prob = action[2], reward = reward(env), terminal = is_terminated(env)))
end

function Base.push!(agent::Agent, ::PostEpisodeStage, env::AbstractEnv)
    if haskey(agent.trajectory, :next_action) 
        action = RLBase.plan!(agent.policy, env)
        push!(agent.trajectory, PartialNamedTuple((action = action, )))
    end
end
