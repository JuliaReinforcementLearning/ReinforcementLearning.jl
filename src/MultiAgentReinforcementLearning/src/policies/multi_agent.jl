export MultiAgentManager

Base.@kwdef mutable struct MultiAgentManager{T <: AbstractPolicy, S} <: AbstractPolicy
    agents::Dict{S,T}
    cur_player::S = first(keys(agents))
end

Base.getindex(A::MultiAgentManager, x) = getindex(A.agents, x)

"""
    MultiAgentManager(player => policy...)

This is the simplest form of multiagent system. At each step they observe the
environment from their own perspective and get updated independently.
"""

RLBase.prob(A::MultiAgentManager, env::AbstractEnv, args...) = prob(A[A.cur_player].policy, env, args...)

(A::MultiAgentManager)(env::AbstractEnv) = A(env, DynamicStyle(env))

function (A::MultiAgentManager)(env::AbstractEnv, ::Sequential)
    while current_player(env) == chance_player(env)
        env |> legal_action_space |> rand |> env
    end
    A.cur_player = current_player(env)
    return A[A.cur_player](env)
end

function (A::MultiAgentManager)(env::AbstractEnv, ::Simultaneous)
    @error "MultiAgentManager doesn't support simultaneous environments. Please consider applying `SequentialEnv` wrapper to environment first."
end

function (A::MultiAgentManager)(stage::PreActStage, env::AbstractEnv)
    A.cur_player = current_player(env)
    A[A.cur_player](stage, env)
end

function (A::MultiAgentManager)(stage::AbstractStage, env::AbstractEnv)
    A[A.cur_player](stage, env)
end

function (A::MultiAgentManager{<:Agent})(::PostActStage, env::AbstractEnv)
    # in the multi agent case, the immediate rewards are updated when last player took its action
    if A.cur_player == last(players(env))
        for (p, agent) in A.agents
            update!(agent.cache, reward(env, p), is_terminated(env))
        end
    end
end

function RLBase.optimise!(A::MultiAgentManager)
    for (_, agent) in A.agents
        RLBase.optimise!(agent)
    end
end
