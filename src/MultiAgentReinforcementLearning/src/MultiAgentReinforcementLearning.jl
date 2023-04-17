module MultiAgentReinforcementLearning

const MultiAgentRL = MultiAgentReinforcementLearning
export MultiAgentRL

export MultiAgentPolicy
export MultiAgentHook

using ReinforcementLearningBase
using ReinforcementLearningCore

import ReinforcementLearningCore: RLCore
import Base.getindex
import Base.iterate

struct MultiAgentPolicy{NT<: NamedTuple} <: AbstractPolicy
    agents::NT

    function MultiAgentPolicy(agents::NamedTuple)
        new(agents)
    end
end

struct MultiAgentHook{NT<: NamedTuple} <: AbstractHook
    hooks::NamedTuple::NT

    function MultiAgentHook(hooks::NamedTuple)
        new(hooks)
    end
end

# (p::MultiAgentPolicy)(env::AbstractEnv) = nothing # Default does nothing, but might be useful for some environments to clean up / pass final state to agents

struct CurrentPlayerIterator{E<:AbstractEnv}
    env::E
end


Base.iterate(current_player_iterator::CurrentPlayerIterator, env = current_player_iterator.env) =
    (current_player(current_player_iterator.env), env)

Base.iterate(p::MultiAgentPolicy) = iterate(p.agents)
Base.iterate(p::MultiAgentPolicy, s) = iterate(p.agents, s)

Base.getindex(p::MultiAgentPolicy, s::Symbol) = p.agents[s]
Base.getindex(h::MultiAgentHook, s::Symbol) = h.hooks[s]

Base.keys(p::MultiAgentPolicy) = keys(p.agents)
Base.keys(p::MultiAgentHook) = keys(p.hooks)

function RLCore._run(
    multiagent_policy::MultiAgentPolicy,
    env::AbstractEnv,
    stop_condition,
    hook::MultiAgentHook,
    reset_condition,
)
    _run(
        multiagent_policy::MultiAgentPolicy,
        env::AbstractEnv,
        DynamicStyle(env), # Dispatch on sequential / simultaneous traits
        stop_condition,
        hook,
        reset_condition,
    )
end


function (multiagent::MultiAgentPolicy)(::PreEpisodeStage, env::AbstractEnv)
    for agent in multiagent
        agent(PreEpisodeStage(), env)
    end
end

function (multiagent::MultiAgentPolicy)(::PreActStage, env::AbstractEnv)
    for agent in multiagent
        agent(PreActStage(), env)
    end
end

function (multiagent::MultiAgentPolicy)(::PostActStage, env::AbstractEnv)
    for agent in multiagent
        agent(PostActStage(), env)
    end
end

function (multiagent::MultiAgentPolicy)(::PostEpisodeStage, env::AbstractEnv)
    for agent in multiagent
        agent(PostEpisodeStage(), env)
    end
end

function (hook::MultiAgentHook)(::PreEpisodeStage, multiagent::MultiAgentPolicy, env::AbstractEnv)
    for (name, agent) in pairs(multiagent)
        hook[name](PreEpisodeStage(), agent, env)
    end
end

function (hook::MultiAgentHook)(::PreActStage, multiagent::MultiAgentPolicy, env::AbstractEnv)
    for (name, agent) in pairs(multiagent)
        hook[name](PreActStage(), agent, env)
    end
end

function (hook::MultiAgentHook)(::PostActStage, multiagent::MultiAgentPolicy, env::AbstractEnv)
    for (name, agent) in pairs(multiagent)
        hook[name](PostActStage(), agent, env)
    end
end

function (hook::MultiAgentHook)(::PostEpisodeStage, multiagent::MultiAgentPolicy, env::AbstractEnv)
    for (name, agent) in pairs(multiagent)
        hook[name](PostEpisodeStage(), agent, env)
    end
end


function (multiagent::MultiAgentPolicy)(env::AbstractEnv)
    return (agent(env) for agent in multiagent)
end

# function RLCore.check(policy::MultiAgentPolicy, env::AbstractEnv)
#     keys(policy) == keys(env
# end

function _run(
    multiagent_policy::MultiAgentPolicy,
    env::AbstractEnv,
    ::Sequential,
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
                    multiagent_policy(env)  # let the policy see the last observation
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

function _run(
    multiagent_policy::MultiAgentPolicy,
    env::AbstractEnv,
    ::Simultaneous,
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
            multiagent_policy(PreActStage(), env)
            hook(PreActStage(), multiagent_policy, env)

            actions = multiagent_policy(env)
            env(actions)

            optimise!(multiagent_policy)

            multiagent_policy(PostActStage(), env)
            hook(PostActStage(), multiagent_policy, env)

            if stop_condition(multiagent_policy, env)
                is_stop = true
                multiagent_policy(PreActStage(), env)
                hook(PreActStage(), multiagent_policy, env)
                multiagent_policy(env)  # let the policy see the last observation
                break
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
