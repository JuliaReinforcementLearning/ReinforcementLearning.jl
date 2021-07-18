export NFSPAgentManager


"""
    Neural Fictitious Self-Play (NFSP) agent implemented in Julia.

See the paper https://arxiv.org/abs/1603.01121 for more details.
"""
mutable struct NFSPAgentManager <: AbstractPolicy
    agents::Dict{Any, NFSPAgent}
end


function NFSPAgentManager(env::AbstractEnv; kwargs...)
    NFSPAgentManager(
        Dict((player, NFSPAgent(env, player; kwargs...)) 
        for player in players(env) if player != chance_player(env)
        )
    )
end


function (π::NFSPAgentManager)(env::AbstractEnv)
    player = current_player(env)
    if player == chance_player(env)
        env |> legal_action_space |> rand |> env
    else
        env |> π.agents[player] |> env
    end
end


RLBase.prob(π::NFSPAgentManager, env::AbstractEnv, args...) = prob(π.agents[current_player(env)], env, args...)


function RLBase.update!(π::NFSPAgentManager, env::AbstractEnv)
    player = current_player(env)
    if player == chance_player(env)
        env |> legal_action_space |> rand |> env
    else
        RLBase.update!(π.agents[player], env)
    end
end