export MultiAgentPolicy
export MultiAgentHook

using Random # for RandomPolicy

import Base.getindex
import Base.iterate

struct MultiAgentPolicy{NT<: NamedTuple} <: AbstractPolicy
    agents::NT

    function MultiAgentPolicy(agents::NT) where {NT<: NamedTuple}
        new{NT}(agents)
    end
end

struct MultiAgentHook{NT<: NamedTuple} <: AbstractHook
    hooks::NT

    function MultiAgentHook(hooks::NT) where {NT<: NamedTuple}
        new{NT}(hooks)
    end
end

# (p::MultiAgentPolicy)(env::AbstractEnv) = nothing # Default does nothing, but might be useful for some environments to clean up / pass final state to agents

struct CurrentPlayerIterator{E<:AbstractEnv}
    env::E
end

Base.iterate(current_player_iterator::CurrentPlayerIterator) =
    (current_player(current_player_iterator.env), current_player_iterator.env)

function Base.iterate(current_player_iterator::CurrentPlayerIterator, state)
    next_player!(current_player_iterator.env)
    return (current_player(current_player_iterator.env), state)
end

Base.iterate(p::MultiAgentPolicy) = iterate(p.agents)
Base.iterate(p::MultiAgentPolicy, s) = iterate(p.agents, s)

Base.getindex(p::MultiAgentPolicy, s::Symbol) = p.agents[s]
Base.getindex(h::MultiAgentHook, s::Symbol) = h.hooks[s]

Base.keys(p::MultiAgentPolicy) = keys(p.agents)
Base.keys(p::MultiAgentHook) = keys(p.hooks)

function Base.run(
    multiagent_policy::MultiAgentPolicy,
    env::E,
    stop_condition,
    hook::H,
    reset_condition,
) where {E<:AbstractEnv, H<:AbstractHook}
    keys(multiagent_policy) == keys(hook) || throw(ArgumentError("MultiAgentPolicy and MultiAgentHook must have the same keys"))
    Base.run(
        multiagent_policy,
        env,
        DynamicStyle(env), # Dispatch on sequential / simultaneous traits
        stop_condition,
        hook,
        reset_condition,
    )
end

function Base.run(
    multiagent_policy::MultiAgentPolicy,
    env::E,
    ::Sequential,
    stop_condition,
    hook,
    reset_condition,
) where {E<:AbstractEnv}
    hook(PreExperimentStage(), multiagent_policy, env)
    multiagent_policy(PreExperimentStage(), env)
    is_stop = false
    while !is_stop
        reset!(env)
        multiagent_policy(PreEpisodeStage(), env)
        hook(PreEpisodeStage(), multiagent_policy, env)

        while !reset_condition(multiagent_policy, env) # one episode
            for player in CurrentPlayerIterator(env)
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

function Base.run(
    multiagent_policy::MultiAgentPolicy,
    env::E,
    ::Simultaneous,
    stop_condition,
    hook,
    reset_condition,
) where {E<:AbstractEnv}
    RLCore._run(
        multiagent_policy,
        env,
        stop_condition,
        hook,
        reset_condition,
    )
end

function (multiagent::MultiAgentPolicy)(::PreEpisodeStage, env::E) where {E<:AbstractEnv}
    for player in players(env)
        multiagent[player](PreEpisodeStage(), env)
    end
end

function (multiagent::MultiAgentPolicy)(::PreActStage, env::E) where {E<:AbstractEnv}
    for player in players(env)
        RLCore.update!(multiagent[player], state(env, player))
    end
end

function (multiagent::MultiAgentPolicy)(::PostActStage, env::E) where {E<:AbstractEnv}
    for player in players(env)
        RLCore.update!(multiagent[player].cache, reward(env, player), is_terminated(env))
    end
end

function (multiagent::MultiAgentPolicy)(::PostEpisodeStage, env::E) where {E<:AbstractEnv}
    for player in players(env)
        multiagent[player](PostEpisodeStage(), env)
    end
end

function (hook::MultiAgentHook)(::PreEpisodeStage, multiagent::MultiAgentPolicy, env::E) where {E<:AbstractEnv}
    for player in players(env)
        hook[player](PreEpisodeStage(), multiagent[player], env)
    end
end

function (hook::MultiAgentHook)(::PreActStage, multiagent::MultiAgentPolicy, env::E) where {E<:AbstractEnv}
    for player in players(env)
        hook[player](PreActStage(), multiagent[player], env)
    end
end

function (hook::MultiAgentHook)(::PostActStage, multiagent::MultiAgentPolicy, env::E) where {E<:AbstractEnv}
    for player in players(env)
        hook[player](PostActStage(), multiagent[player], env)
    end
end

function (hook::MultiAgentHook)(::PostEpisodeStage, multiagent::MultiAgentPolicy, env::E) where {E<:AbstractEnv}
    for player in players(env)
        hook[player](PostEpisodeStage(), multiagent[player], env)
    end
end

function (multiagent::MultiAgentPolicy)(env::E) where {E<:AbstractEnv}
    return (multiagent[player](env, player) for player in players(env))
end
