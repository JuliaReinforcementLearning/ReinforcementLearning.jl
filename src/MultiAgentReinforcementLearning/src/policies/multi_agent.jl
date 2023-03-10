export MultiAgentManager

struct MultiAgentManager <: AbstractPolicy
    agents::Dict{Any,Any}
end

Base.getindex(A::MultiAgentManager, x) = getindex(A.agents, x)

"""
    MultiAgentManager(player => policy...)

This is the simplest form of multiagent system. At each step they observe the
environment from their own perspective and get updated independently. For
environments of `SEQUENTIAL` style, agents which are not the current player will
observe a dummy action of [`NO_OP`](@ref) in the `PreActStage`. For environments
of `SIMULTANEOUS` style, please wrap it with [`SequentialEnv`](@ref) first.
"""
MultiAgentManager(policies...) =
    MultiAgentManager(Dict{Any,Any}(nameof(p) => p for p in policies))

RLBase.prob(A::MultiAgentManager, env::AbstractEnv, args...) = prob(A[current_player(env)].policy, env, args...)

(A::MultiAgentManager)(env::AbstractEnv) = A(env, DynamicStyle(env))

(A::MultiAgentManager)(env::AbstractEnv, ::Sequential) = A[current_player(env)](env)

function (A::MultiAgentManager)(env::AbstractEnv, ::Simultaneous)
    @error "MultiAgentManager doesn't support simultaneous environments. Please consider applying `SequentialEnv` wrapper to environment first."
end

function (A::MultiAgentManager)(stage::AbstractStage, env::AbstractEnv)
    for agent in values(A.agents)
        agent(stage, env)
    end
end

function RLBase.optimise!(A::MultiAgentManager)
    for agent in values(A.agents)
        RLBase.optimise!(agent)
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
