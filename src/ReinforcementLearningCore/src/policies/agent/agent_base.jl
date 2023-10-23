export Agent, OfflineAgent

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

mutable struct OfflineAgent{P,T,B,R} <: AbstractPolicy
    policy::P
    trajectory::T
    behavior_agent::B #a behavior agent to fill the trajectory. Leave nothing if the trajectory is prefilled. Should share the same trajectory as the parent OfflineAgent
    behavior_steps::Int #steps to fill the trajectory (defaults to capacity of )
    behavior_reset_condition::R #the reset condition of the environment.
    function OfflineAgent(policy::P, trajectory::T, behavior_agent = nothing, behavior_steps = ReinforcementLearningTrajectories.capacity(trajectory.container.traces), behavior_reset_condition = ResetAtTerminal()) where {P<:AbstractPolicy, T<:Trajectory}
        if behavior_steps == Inf
            @error "`behavior_steps` is infinite, please provide a finite integer."
        end
        agent = new{P,T, typeof(behavior_agent), typeof(behavior_reset_condition)}(policy, trajectory, behavior_agent, behavior_steps, behavior_reset_condition)
        if TrajectoryStyle(trajectory) === AsyncTrajectoryStyle()
            bind(trajectory, @spawn(optimise!(policy, trajectory)))
        end
        agent
    end
end

Agent(;policy, trajectory) = Agent(policy, trajectory)

RLBase.optimise!(agent::Union{Agent,OfflineAgent}, stage::S) where {S<:AbstractStage} = RLBase.optimise!(TrajectoryStyle(agent.trajectory), agent, stage)
RLBase.optimise!(::SyncTrajectoryStyle, agent::Union{Agent,OfflineAgent}, stage::S) where {S<:AbstractStage} = RLBase.optimise!(agent.policy, stage, agent.trajectory)

# already spawn a task to optimise inner policy when initializing the agent
RLBase.optimise!(::AsyncTrajectoryStyle, agent::Union{Agent,OfflineAgent}, stage::S) where {S<:AbstractStage} = nothing

#by default, optimise does nothing at all stage
function RLBase.optimise!(policy::AbstractPolicy, stage::AbstractStage, trajectory::Trajectory) end

@functor Agent (policy,)

function Base.push!(agent::Agent, ::PreEpisodeStage, env::AbstractEnv)
    push!(agent.trajectory, (state = state(env),))
end

# !!! TODO: In async scenarios, parameters of the policy may still be updating
# (partially), which will result to incorrect action. This should be addressed
# in Oolong.jl with a wrapper
function RLBase.plan!(agent::Union{Agent,OfflineAgent}, env::AbstractEnv)
    RLBase.plan!(agent.policy, env)
end

function Base.push!(agent::Agent, ::PostActStage, env::AbstractEnv, action)
    next_state = state(env)
    push!(agent.trajectory, (state = next_state, action = action, reward = reward(env), terminal = is_terminated(env)))
end

function Base.push!(agent::Agent, ::PostEpisodeStage, env::AbstractEnv)
    if haskey(agent.trajectory, :next_action) 
        action = RLBase.plan!(agent.policy, env)
        push!(agent.trajectory, PartialNamedTuple((action = action, )))
    end
end

OfflineAgent(;policy, trajectory, behavior_agent=nothing, behavior_steps = ReinforcementLearningTrajectories.capacity(trajectory.container.traces), behavior_reset_condition = ResetAtTerminal()) = OfflineAgent(policy, trajectory, behavior_agent, behavior_steps, behavior_reset_condition)
@functor OfflineAgent (policy,)

Base.push!(::OfflineAgent{P,T,B}, ::PreExperimentStage, env) where {P,T,B <: Nothing} = nothing
#fills the trajectory with interactions generated with the behavior_agent at the PreExperimentStage.
function Base.push!(agent::OfflineAgent, ::PreExperimentStage, env::AbstractEnv)
    is_stop = false
    policy = agent.behavior_agent
    steps = 0
    while !is_stop
        steps += 1
        reset!(env)
        push!(policy, PreEpisodeStage(), env)

        while !agent.behavior_reset_condition(policy, env) # one episode
            push!(policy, PreActStage(), env)
            action = RLBase.plan!(policy, env)
            act!(env, action)
            push!(policy, PostActStage(), env, action)

            if steps >= agent.behavior_steps
                is_stop = true
                break
            end
        end # end of an episode
    push!(policy, PostEpisodeStage(), env)
    end    
end