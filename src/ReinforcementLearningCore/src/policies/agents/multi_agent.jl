export MultiAgentManager, NO_OP, NoOp

"Represent no-operation if it's not the agent's turn."
struct NoOp end

const NO_OP = NoOp()

struct MultiAgentManager <: AbstractPolicy
    agents::Dict{Any,Any}
end

Base.getindex(A::MultiAgentManager, x) = getindex(A.agents, x)

"""
    MultiAgentManager(player => policy...)

This is the simplest form of multiagent system. At each step they observe the
environment from their own perspective and get updated independently. For
environments of `SEQUENTIAL` style, agents which are not the current player will
observe a dummy action of [`NO_OP`](@ref) in the `PreActStage`.
"""
MultiAgentManager(policies...) =
    MultiAgentManager(Dict{Any,Any}(nameof(p) => p for p in policies))

(A::MultiAgentManager)(env::AbstractEnv) = A(env, DynamicStyle(env))
(A::MultiAgentManager)(env::AbstractEnv, ::Sequential) = A[current_player(env)](env)
(A::MultiAgentManager)(env::AbstractEnv, ::Simultaneous) =
    [agent(env) for agent in values(A.agents)]

function (A::MultiAgentManager)(stage::AbstractStage, env::AbstractEnv)
    for agent in values(A.agents)
        agent(stage, env)
    end
end

function (A::MultiAgentManager)(stage::PreActStage, env::AbstractEnv, action)
    A(stage, env, DynamicStyle(env), action)
end

function (A::MultiAgentManager)(stage::PreActStage, env::AbstractEnv, ::Sequential, action)
    p = current_player(env)
    for (player, agent) in A.agents
        if p == player
            agent(stage, env, action)
        else
            agent(stage, env, NO_OP)
        end
    end
end

function (A::MultiAgentManager)(
    stage::PreActStage,
    env::AbstractEnv,
    ::Simultaneous,
    actions,
)
    for (agent, action) in zip(values(A.agents), actions)
        agent(stage, env, action)
    end
end
