module MultiAgentReinforcementLearning

export MultiAgentPolicy

using ReinforcementLearningBase
using ReinforcementLearningCore

import ReinforcementLearningCore: RLCore
import Base.getindex
import Base.iterate

struct MultiAgentPolicy <: AbstractPolicy
    agents::NamedTuple

    function MultiAgentPolicy(agents::NamedTuple)
        new(agents)
    end
end

struct CurrentPlayerIterator
    env::AbstractEnv
end

function current_player_iterator(env)
    return CurrentPlayerIterator(env)
end

Base.iterate(current_player_iterator::CurrentPlayerIterator) =
    (current_player(current_player_iterator.env), current_player_iterator.env)

Base.iterate(current_player_iterator::CurrentPlayerIterator, env) =
    (current_player(current_player_iterator.env), current_player_iterator.env)

Base.getindex(p::MultiAgentPolicy, s::Symbol) = p.agents[s]

function RLCore._run(
    multiagent_policy::MultiAgentPolicy,
    env::AbstractEnv,
    stop_condition,
    hook,
    reset_condition,
)

    hook(PreExperimentStage(), multiagent_policy, env)
    multiagent_policy(PreExperimentStage(), env)
    is_stop = false
    while !is_stop
        reset!(env)
        multiagent_policy(PreEpisodeStage(), env)
        hook(PreEpisodeStage(), multiagent_policy, env)

        while !reset_condition(multiagent_policy, env) # one episode
            for player in current_player_iterator(env)
                policy = multiagent_policy[player] # Select appropriate policy

                policy(PreActStage(), env)
                hook(PreActStage(), policy, env)

                action = policy(env)
                env(action)

                optimise!(policy)

                policy(PostActStage(), env)
                hook(PostActStage(), policy, env)

                if stop_condition(policy, env)
                    is_stop = true
                    policy(PreActStage(), env)
                    hook(PreActStage(), policy, env)
                    policy(env)  # let the policy see the last observation
                    break
                end
            end
        end # end of an episode

        if is_terminated(env)
            multiagent_policy(PostEpisodeStage(), env)  # let the policy see the last observation
            hook(PostEpisodeStage(), multiagent_policy, env)
        end
    end
    multiagent_policy(PostExperimentStage(), env)
    hook(PostExperimentStage(), multiagent_policy, env)
    hook
end

end # module
