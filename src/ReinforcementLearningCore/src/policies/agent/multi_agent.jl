export MultiAgentPolicy
export MultiAgentHook

using Random # for RandomPolicy

import Base.getindex
import Base.iterate

"""
    MultiAgentPolicy(agents::NT) where {NT<: NamedTuple}

MultiAgentPolicy is a policy struct that contains `<:AbstractPolicy` structs indexed by the player's symbol.
"""
struct MultiAgentPolicy{NT<: NamedTuple} <: AbstractPolicy
    agents::NT

    function MultiAgentPolicy(agents::NT) where {NT<: NamedTuple}
        new{NT}(agents)
    end
end

"""
    MultiAgentHook(hooks::NT) where {NT<: NamedTuple}

MultiAgentHook is a hook struct that contains `<:AbstractoHook` structs indexed by the player's symbol.
"""
struct MultiAgentHook{NT<: NamedTuple} <: AbstractHook
    hooks::NT

    function MultiAgentHook(hooks::NT) where {NT<: NamedTuple}
        new{NT}(hooks)
    end
end

"""
    CurrentPlayerIterator(env::E) where {E<:AbstractEnv}

`CurrentPlayerIterator`` is an iterator that iterates over the players in the environment, returning the `current_player`` for each iteration. This is only necessary for `MultiAgent` environments. After each iteration, `RLBase.next_player!` is called to advance the `current_player`. As long as ``RLBase.next_player!` is defined for the environment, this iterator will work correctly in the `Base.run`` function.
"""
struct CurrentPlayerIterator{E<:AbstractEnv}
    env::E
end

Base.iterate(current_player_iterator::CurrentPlayerIterator) =
    (current_player(current_player_iterator.env), current_player_iterator.env)

function Base.iterate(current_player_iterator::CurrentPlayerIterator, state)
    RLBase.next_player!(current_player_iterator.env)
    return (current_player(current_player_iterator.env), state)
end

Base.iterate(p::MultiAgentPolicy) = iterate(p.agents)
Base.iterate(p::MultiAgentPolicy, s) = iterate(p.agents, s)

Base.getindex(p::MultiAgentPolicy, s::Symbol) = p.agents[s]
Base.getindex(h::MultiAgentHook, s::Symbol) = h.hooks[s]

Base.keys(p::MultiAgentPolicy) = keys(p.agents)
Base.keys(p::MultiAgentHook) = keys(p.hooks)


"""
    Base.run(
        multiagent_policy::MultiAgentPolicy,
        env::E,
        stop_condition,
        hook::MultiAgentHook,
        reset_condition,
    ) where {E<:AbstractEnv, H<:AbstractHook}

This run function dispatches games using `MultiAgentPolicy` and `MultiAgentHook` to the appropriate `run` function based on the `Sequential` or `Simultaneous` trait of the environment.
"""
function Base.run(
    multiagent_policy::MultiAgentPolicy,
    env::E,
    stop_condition,
    hook::MultiAgentHook,
    reset_condition,
) where {E<:AbstractEnv}
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

"""
    Base.run(
        multiagent_policy::MultiAgentPolicy,
        env::E,
        ::Sequential,
        stop_condition,
        hook::MultiAgentHook,
        reset_condition,
    ) where {E<:AbstractEnv, H<:AbstractHook}

This run function handles `MultiAgent` games with the `Sequential` trait. It iterates over the `current_player` for each turn in the environment, and runs the full `run` loop, like in the `SingleAgent` case. If the `stop_condition` is met, the function breaks out of the loop and calls `optimise!` on the policy again. Finally, it calls `optimise!` on the policy one last time and returns the `MultiAgentHook`.
"""
function Base.run(
    multiagent_policy::MultiAgentPolicy,
    env::E,
    ::Sequential,
    stop_condition,
    hook::MultiAgentHook,
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


"""
    Base.run(
        multiagent_policy::MultiAgentPolicy,
        env::E,
        ::Simultaneous,
        stop_condition,
        hook::MultiAgentHook,
        reset_condition,
    ) where {E<:AbstractEnv, H<:AbstractHook}

This run function handles `MultiAgent` games with the `Simultaneous` trait. It iterates over the players in the environment, and for each player, it selects the appropriate policy from the `MultiAgentPolicy`. All agent actions are collected before the environment is updated. After each player has taken an action, it calls `optimise!` on the policy. If the `stop_condition` is met, the function breaks out of the loop and calls `optimise!` on the policy again. Finally, it calls `optimise!` on the policy one last time and returns the `MultiAgentHook`.
"""
function Base.run(
    multiagent_policy::MultiAgentPolicy,
    env::E,
    ::Simultaneous,
    stop_condition,
    hook::MultiAgentHook,
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
