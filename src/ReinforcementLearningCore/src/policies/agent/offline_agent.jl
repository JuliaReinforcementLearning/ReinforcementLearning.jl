export OfflineAgent, OfflineBehavior

using Flux

"""
    OfflineBehavior(; agent:: Union{<:Agent, Nothing}, steps::Int, reset_condition)

Used to provide an OfflineAgent with a "behavior agent" that will generate the training data
at the `PreExperimentStage`. If `agent` is `nothing` (by default), does nothing. The `trajectory` of agent should 
be the same as that of the parent `OfflineAgent`.
`steps` is the number of data elements to generate, defaults to the capacity of the trajectory.
`reset_condition` is the episode reset condition for the data generation (defaults to `ResetIfEnvTerminated()`).

The behavior agent will interact with the main environment of the experiment to generate the data.
"""
struct OfflineBehavior{A<:Union{<:Agent,Nothing},R}
    agent::A
    steps::Int
    reset_condition::R
end

OfflineBehavior() = OfflineBehavior(nothing, 0, ResetIfEnvTerminated())

function OfflineBehavior(agent; steps=ReinforcementLearningTrajectories.capacity(agent.trajectory.container.traces), reset_condition=ResetIfEnvTerminated())
    if steps == Inf
        @error "`steps` is infinite, please provide a finite integer."
    end
    OfflineBehavior(agent, steps, reset_condition)
end

"""
    OfflineAgent(policy::AbstractPolicy, trajectory::Trajectory, offline_behavior::OfflineBehavior = OfflineBehavior()) <: AbstractAgent

`OfflineAgent` is an `AbstractAgent` that, unlike the usual online `Agent`, does not interact with the environment
during training in order to collect data. Just like `Agent`, it contains an `AbstractPolicy` to be trained an a `Trajectory`
that contains the training data. The difference being that the trajectory is filled prior to training and is not updated.
An `OfflineBehavior` can optionally be provided to provide an second "behavior agent" that will
generate the training data at the `PreExperimentStage`. Does nothing by default. 
"""
struct OfflineAgent{P<:AbstractPolicy,T<:Trajectory,B<:OfflineBehavior} <: AbstractAgent
    policy::P
    trajectory::T
    offline_behavior::B
    function OfflineAgent(policy::P, trajectory::T, offline_behavior=OfflineBehavior()) where {P<:AbstractPolicy,T<:Trajectory}
        agent = new{P,T,typeof(offline_behavior)}(policy, trajectory, offline_behavior)
        if TrajectoryStyle(trajectory) === AsyncTrajectoryStyle()
            bind(trajectory, @spawn(optimise!(policy, trajectory)))
        end
        agent
    end
end

OfflineAgent(; policy, trajectory, offline_behavior=OfflineBehavior()) = OfflineAgent(policy, trajectory, offline_behavior)
Flux.@layer OfflineAgent trainable=(policy,)

Base.push!(::OfflineAgent{P,T,<:OfflineBehavior{Nothing}}, ::PreExperimentStage, env::AbstractEnv) where {P,T} = nothing
#fills the trajectory with interactions generated with the behavior_agent at the PreExperimentStage.
function Base.push!(agent::OfflineAgent{P,T,<:OfflineBehavior{<:Agent}}, ::PreExperimentStage, env::AbstractEnv) where {P,T}
    is_stop = false
    policy = agent.offline_behavior.agent
    steps = 0
    while !is_stop
        reset!(env)
        push!(policy, PreEpisodeStage(), env)
        while !check!(agent.offline_behavior.reset_condition, policy, env) # one episode
            steps += 1
            push!(policy, PreActStage(), env)
            action = RLBase.plan!(policy, env)
            act!(env, action)
            push!(policy, PostActStage(), env, action)
            if steps >= agent.offline_behavior.steps
                is_stop = true
                break
            end
        end # end of an episode
        push!(policy, PostEpisodeStage(), env)
    end
end
