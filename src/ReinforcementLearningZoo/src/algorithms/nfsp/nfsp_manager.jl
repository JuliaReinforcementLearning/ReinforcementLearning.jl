export NFSPAgentManager

## definition
"""
    NFSPAgentManager(; agents::Dict{Any, NFSPAgent})

A special MultiAgentManager in which all agents use NFSP policy to play the game.
"""
mutable struct NFSPAgentManager <: AbstractPolicy
    agents::Dict{Any, NFSPAgent}
end

## interactions between the policy and env.
function (π::NFSPAgentManager)(env::AbstractEnv)
    player = current_player(env)
    if player == chance_player(env)
        env |> legal_action_space |> rand
    else
        env |> π.agents[player]
    end
end

RLBase.prob(π::NFSPAgentManager, env::AbstractEnv, args...) = prob(π.agents[current_player(env)], env, args...)

## update NFSPAgentManager
function RLBase.update!(π::NFSPAgentManager, env::AbstractEnv)
    while current_player(env) == chance_player(env)
        env |> legal_action_space |> rand |> env
    end
    update!(π.agents[current_player(env)], env)
end

function (π::NFSPAgentManager)(stage::Union{PreEpisodeStage, PostEpisodeStage}, env::AbstractEnv)
    @sync for (player, agent) in π.agents
        @async agent(stage, env, player)
    end
end
