export MultiAgentManager

Base.@kwdef mutable struct MultiAgentManager{T <: AbstractPolicy} <: AbstractPolicy
    agents::Dict{S,T} where {S<: Union{String, Symbol, Integer}}
    cur_player::Union{String, Symbol, Integer} = first(keys(agents))
end

Base.getindex(A::MultiAgentManager, x) = getindex(A.agents, x)

"""
    MultiAgentManager(player => policy...)

This is the simplest form of multiagent system. At each step they observe the
environment from their own perspective and get updated independently.
"""
MultiAgentManager(policies...) =
    MultiAgentManager(Dict{Any,Any}(nameof(p) => p for p in policies))
  
RLBase.prob(A::MultiAgentManager, env::AbstractEnv, args...) = prob(A[A.cur_player].policy, env, args...)

(A::MultiAgentManager)(env::AbstractEnv) = A(env, DynamicStyle(env))

function (A::MultiAgentManager)(env::AbstractEnv, ::Sequential)
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

function RLBase.optimise!(A::MultiAgentManager)
    for (_, agent) in A.agents
        RLBase.optimise!(agent)
    end
end
