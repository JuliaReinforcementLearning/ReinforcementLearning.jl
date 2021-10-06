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
  4. In the `PosEpisodeStage`, we push the `state` at the end of an episode and
     a dummy action into the `trajectory`.
  5. In the `PreEpisodeStage`, we pop out the latest `state` and `action` pair
     (which are dummy ones) from `trajectory`.

2. Update the inner `policy` given the context of `trajectory`, `env`, and
   `stage`.
  1. By default, we only `update!` the `policy` in the `PreActStage`. And it's
     dispatched to `update!(policy, trajectory, env, stage)`.
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
    trajectory::AbstractTrajectory,
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
    trajectory::AbstractTrajectory,
    policy::AbstractPolicy,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    s = policy isa NamedPolicy ? state(env, nameof(policy)) : state(env)
    push!(trajectory[:state], s)
    push!(trajectory[:action], action)
    if haskey(trajectory, :legal_actions_mask)
        lasm =
            policy isa NamedPolicy ? legal_action_space_mask(env, nameof(policy)) :
            legal_action_space_mask(env)
        push!(trajectory[:legal_actions_mask], lasm)
    end
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    policy::AbstractPolicy,
    env::AbstractEnv,
    ::PostEpisodeStage,
)
    # Note that for trajectories like `CircularArraySARTTrajectory`, data are
    # stored in a SARSA format, which means we still need to generate a dummy
    # action at the end of an episode. Here we simply select a random one using
    # the global rng. In theory it shouldn't affect the performance of specific
    # algorithm.
    # TODO: how to inject a local rng here to avoid polluting the global rng

    s = policy isa NamedPolicy ? state(env, nameof(policy)) : state(env)
    a =
        policy isa NamedPolicy ? rand(action_space(env, nameof(policy))) :
        rand(action_space(env))
    push!(trajectory[:state], s)
    push!(trajectory[:action], a)
    if haskey(trajectory, :legal_actions_mask)
        lasm =
            policy isa NamedPolicy ? legal_action_space_mask(env, nameof(policy)) :
            legal_action_space_mask(env)
        push!(trajectory[:legal_actions_mask], lasm)
    end
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    policy::AbstractPolicy,
    env::AbstractEnv,
    ::PostActStage,
)
    r = policy isa NamedPolicy ? reward(env, nameof(policy)) : reward(env)
    push!(trajectory[:reward], r)
    push!(trajectory[:terminal], is_terminated(env))
end

#####
# Pre-training
#####
function (agent::Agent)(stage::PreExperimentStage, env::AbstractEnv)
    update!(agent.policy, agent.trajectory, env, stage)
end
