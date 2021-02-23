export DynaAgent

Base.@kwdef struct DynaAgent{P,M,T} <: AbstractPolicy
    policy::P
    model::M
    trajectory::T
    plan_step::Int = 10
end

(p::DynaAgent)(env::AbstractEnv) = p.policy(env)

function (agent::DynaAgent)(stage::AbstractStage, env::AbstractEnv)
    update!(agent.trajectory, agent.policy, env, stage)
    dyna_update!(agent, env, stage)
end

function (agent::DynaAgent)(stage::PreActStage, env::AbstractEnv, action)
    update!(agent.trajectory, agent.policy, env, stage, action)
    dyna_update!(agent, env, stage)
end

function dyna_update!(agent, env, stage)
    # 1. model learning
    update!(agent.model, agent.trajectory, agent.policy, env, stage)
    # 2. direct learning
    update!(agent.policy, agent.trajectory, env, stage)
    # 3. policy learning
    for _ in 1:agent.plan_step
        update!(agent.policy, agent.model, agent.trajectory, env, stage)
    end
end

# 1. model learning
# By default we do nothing
function RLBase.update!(
    ::AbstractEnvironmentModel,
    ::AbstractTrajectory,
    ::AbstractPolicy,
    ::AbstractEnv,
    ::AbstractStage,
) end

# 3. policy learning
# By default we do nothing
function RLBase.update!(
    ::AbstractPolicy,
    ::AbstractEnvironmentModel,
    ::AbstractTrajectory,
    ::AbstractEnv,
    ::AbstractStage,
) end
