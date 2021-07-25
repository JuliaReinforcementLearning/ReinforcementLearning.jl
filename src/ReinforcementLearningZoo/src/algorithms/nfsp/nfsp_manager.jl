export NFSPAgentManager


"""
    NFSPAgentManager(; agents::Dict{Any, NFSPAgent})

A special MultiAgentManager in which all agents use NFSP policy to play the game.
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

function (π::NFSPAgentManager)(stage::PostEpisodeStage, env::AbstractEnv)
    for player in players(env)
        if player != chance_player(env)
            π.agents[player](stage, env)
        end
    end
end

function RLBase.prob(π::NFSPAgentManager, env::AbstractEnv, args...)
    agent = π.agents[current_player(env)].sl_agent
    prob(agent.policy, env, args...)
end

function RLBase.update!(π::NFSPAgentManager, env::AbstractEnv)
    player = current_player(env)
    if player == chance_player(env)
        env |> legal_action_space |> rand |> env
    else
        RLBase.update!(π.agents[player], env)
    end
end
