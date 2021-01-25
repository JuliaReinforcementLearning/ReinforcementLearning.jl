export Agent

import Functors: functor
using Setfield: @set

"""
    Agent(;kwargs...)

A wrapper of an `AbstractPolicy`. Generally speaking, it does nothing but to
update the trajectory and policy appropriately in different stages.

# Keywords & Fields

- `policy`::[`AbstractPolicy`](@ref): the policy to use
- `trajectory`::[`AbstractTrajectory`](@ref): used to store transitions between an agent and an environment
"""
Base.@kwdef struct Agent{P<:AbstractPolicy,T<:AbstractTrajectory} <: AbstractPolicy
    policy::P
    trajectory::T
end

functor(x::Agent) = (policy = x.policy,), y -> @set x.policy = y.policy

(agent::Agent)(env) = agent.policy(env)

function check(agent::Agent, env::AbstractEnv)
    if ActionStyle(env) === FULL_ACTION_SET &&
       !haskey(agent.trajectory, :legal_actions_mask)
    #     @warn "The env[$(nameof(env))] is of FULL_ACTION_SET, but I can not find a trace named :legal_actions_mask in the trajectory"
    end
    check(agent.policy, env)
end

Base.nameof(agent::Agent) = nameof(agent.policy)

#####
# Default behaviors
#####

"""
Here we extend the definition of `(p::AbstractPolicy)(::AbstractEnv)` in
`RLBase` to accept an `AbstractStage` as the first argument. Algorithm designers
may customize these behaviors respectively by implementing:

- `(p::YourPolicy)(::AbstractStage, ::AbstractEnv)`
- `(p::YourPolicy)(::PreActStage, ::AbstractEnv, action)`

The default behaviors for `Agent` are:

1. Update the inner `trajectory` given the context of `policy`, `env`, and
   `stage`.
  1. By default we do nothing.
  2. In `PreActStage`, we `push!` the current **state** and the **action** into
     the `trajectory`.
  3. In `PostActStage`, we query the `reward` and `is_terminated` info from
     `env` and push them into `trajectory`.
  4. For `CircularSARTTrajectory`:
     1. In the `PosEpisodeStage`, we push the `state` at the end of an episode
        and a dummy action into the `trajectory`.
     1. In the `PreEpisodeStage`, we pop out the lastest `state` and `action`
        pair (which are dummy ones) from `trajectory`.
2. Update the inner `policy` given the context of `trajectory`, `env`, and
   `stage`.
  1. By default, we only `update!` the `policy` in the `PreActStage`. And it's
     despatched to `update!(policy, trajectory)`.
"""
function (agent::Agent)(stage::AbstractStage, env::AbstractEnv)
    update!(agent.trajectory, agent.policy, env, stage)
    update!(agent.policy, agent.trajectory, env, stage)
end

function (agent::Agent)(stage::PreActStage, env::AbstractEnv, action)
    update!(agent.trajectory, agent.policy, env, stage, action)
    update!(agent.policy, agent.trajectory, env, stage)
end

function RLBase.update!(
    ::AbstractPolicy,
    ::AbstractTrajectory,
    ::AbstractEnv,
    ::AbstractStage,
) end

#####
# Default behaviors for known trajectories
#####

function RLBase.update!(
    ::AbstractTrajectory,
    ::AbstractPolicy,
    ::AbstractEnv,
    ::AbstractStage,
) end

function RLBase.update!(
    trajectory::Union{
        CircularArraySARTTrajectory,
        CircularArraySLARTTrajectory,
        PrioritizedTrajectory{<:CircularArraySARTTrajectory},
        PrioritizedTrajectory{<:CircularArraySLARTTrajectory},
    },
    ::AbstractPolicy,
    ::AbstractEnv,
    ::PreEpisodeStage,
)
    if length(trajectory) > 0
        pop!(trajectory[:state])
        pop!(trajectory[:action])
        if haskey(trajectory, :legal_actions_mask)
            pop!(trajectory[:legal_actions_mask])
        end
    end
end

function RLBase.update!(
    trajectory::Union{
        VectorSARTTrajectory,
        CircularArraySARTTrajectory,
        CircularArraySLARTTrajectory,
        PrioritizedTrajectory{<:CircularArraySARTTrajectory},
        PrioritizedTrajectory{<:CircularArraySLARTTrajectory},
    },
    policy::AbstractPolicy,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    push!(trajectory[:state], state(env))
    push!(trajectory[:action], action)
    if haskey(trajectory, :legal_actions_mask)
        push!(trajectory[:legal_actions_mask], legal_action_space_mask(env))
    end
end

function RLBase.update!(
    trajectory::Union{
        VectorSARTTrajectory,
        CircularArraySARTTrajectory,
        CircularArraySLARTTrajectory,
        PrioritizedTrajectory{<:CircularArraySARTTrajectory},
        PrioritizedTrajectory{<:CircularArraySLARTTrajectory},
    },
    policy::NamedPolicy,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    push!(trajectory[:state], state(env, nameof(policy)))
    push!(trajectory[:action], action)
    if haskey(trajectory, :legal_actions_mask)
        push!(trajectory[:legal_actions_mask], legal_action_space_mask(env, nameof(policy)))
    end
end

function RLBase.update!(
    trajectory::Union{
        VectorSARTTrajectory,
        CircularArraySARTTrajectory,
        CircularArraySLARTTrajectory,
        PrioritizedTrajectory{<:CircularArraySARTTrajectory},
        PrioritizedTrajectory{<:CircularArraySLARTTrajectory},
    },
    policy::AbstractPolicy,
    env::AbstractEnv,
    ::PostEpisodeStage,
)
    # Note that for trajectories like `CircularArraySARTTrajectory`, data are
    # stored in a SARSA format, which means we still need to generate a dummy
    # action at the end of an episode. Here we simply select a random one using
    # the global rng. In theory it shouldn't affect the performance of specific algorithm.
    action = rand(action_space(env))

    push!(trajectory[:state], state(env))
    push!(trajectory[:action], action)
    if haskey(trajectory, :legal_actions_mask)
        push!(trajectory[:legal_actions_mask], legal_action_space_mask(env))
    end
end

function RLBase.update!(
    trajectory::Union{
        VectorSARTTrajectory,
        CircularArraySARTTrajectory,
        CircularArraySLARTTrajectory,
        PrioritizedTrajectory{<:CircularArraySARTTrajectory},
        PrioritizedTrajectory{<:CircularArraySLARTTrajectory},
    },
    policy::NamedPolicy,
    env::AbstractEnv,
    ::PostEpisodeStage,
)
    action = rand(action_space(env))
    push!(trajectory[:state], state(env, nameof(policy)))
    push!(trajectory[:action], action)
    if haskey(trajectory, :legal_actions_mask)
        push!(trajectory[:legal_actions_mask], legal_action_space_mask(env, nameof(policy)))
    end
end


function RLBase.update!(
    trajectory::AbstractTrajectory,
    ::AbstractPolicy,
    env::AbstractEnv,
    ::PostActStage,
)
    push!(trajectory[:reward], reward(env))
    push!(trajectory[:terminal], is_terminated(env))
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    policy::NamedPolicy,
    env::AbstractEnv,
    ::PostActStage,
)
    push!(trajectory[:reward], reward(env, nameof(policy)))
    push!(trajectory[:terminal], is_terminated(env))
end
