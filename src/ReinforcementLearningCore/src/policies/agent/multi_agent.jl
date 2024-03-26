export MultiAgentPolicy, MultiAgentHook, Player, PlayerTuple

using Random # for RandomPolicy

import Base.getindex
import Base.iterate
import Base.push!

"""
    PlayerTuple

A NamedTuple that maps players to their respective values.
"""
struct PlayerTuple{N,T}
    data::NamedTuple{N,T}

    function PlayerTuple(data::Pair...)
        nt = NamedTuple(first(item).name => last(item) for item in data)
        new{typeof(nt).parameters...}(nt)
    end

    function PlayerTuple(data::Base.Generator)
        PlayerTuple(collect(data)...)
    end    
end

Base.getindex(nt::PlayerTuple, player::Player) = nt.data[player.name]
Base.keys(nt::PlayerTuple) = Player.(keys(nt.data))
Base.iterate(nt::PlayerTuple) = iterate(nt.data)
Base.iterate(nt::PlayerTuple, state) = iterate(nt.data, state)

"""
    MultiAgentPolicy(agents::NT) where {NT<: NamedTuple}
MultiAgentPolicy is a policy struct that contains `<:AbstractPolicy` structs indexed by the player's symbol.
"""
struct MultiAgentPolicy{players,T} <: AbstractPolicy
    agents::PlayerTuple{players, T}

    function MultiAgentPolicy(agents::PlayerTuple{players,T}) where {players,T}
        new{players, T}(agents)
    end
end

"""
    MultiAgentHook(hooks::NT) where {NT<: NamedTuple}
MultiAgentHook is a hook struct that contains `<:AbstractHook` structs indexed by the player's symbol.
"""
struct MultiAgentHook{players,T} <: AbstractHook
    hooks::PlayerTuple{players,T}

    function MultiAgentHook(hooks::PlayerTuple{players,T}) where {players, T}
        new{players,T}(hooks)
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
Base.iterate(p::MultiAgentPolicy, state) = iterate(p.agents, state)

Base.getindex(p::MultiAgentPolicy, player::Player) = p.agents[player]
Base.getindex(h::MultiAgentHook, player::Player) = h.hooks[player]

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
    stop_condition::AbstractStopCondition,
    hook::MultiAgentHook,
    reset_condition::AbstractResetCondition=ResetIfEnvTerminated()
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
    stop_condition::AbstractStopCondition,
    multiagent_hook::MultiAgentHook,
    reset_condition::AbstractResetCondition=ResetIfEnvTerminated(),
) where {E<:AbstractEnv}
    push!(multiagent_hook, PreExperimentStage(), multiagent_policy, env)
    push!(multiagent_policy, PreExperimentStage(), env)
    is_stop = false
    while !is_stop
        # NOTE: @timeit_debug statements are for debug logging
        @timeit_debug timer "reset!"                             reset!(env)
        @timeit_debug timer "push!(policy) PreEpisodeStage"      push!(multiagent_policy, PreEpisodeStage(), env)
        @timeit_debug timer "optimise! PreEpisodeStage"          optimise!(multiagent_policy, PreEpisodeStage())
        @timeit_debug timer "push!(hook) PreEpisodeStage"        push!(multiagent_hook, PreEpisodeStage(), multiagent_policy, env)

        while !check!(reset_condition, multiagent_policy, env) && !is_stop # one episode
            for player in CurrentPlayerIterator(env)
                policy = multiagent_policy[player] # Select appropriate policy
                hook = multiagent_hook[player] # Select appropriate hook
                @timeit_debug timer "push!(policy) PreActStage"    push!(policy, PreActStage(), env)
                @timeit_debug timer "optimise! PreActStage"        optimise!(policy, PreActStage())
                @timeit_debug timer "push!(hook) PreActStage"      push!(hook, PreActStage(), policy, env)
                
                action = @timeit_debug timer "plan!"               RLBase.plan!(policy, env)
                @timeit_debug timer "act!" act!(env, action)

                @timeit_debug timer "push!(policy) PostActStage"     push!(policy, PostActStage(), env, action)
                @timeit_debug timer "optimise! PostActStage"         optimise!(policy, PostActStage())
                @timeit_debug timer "push!(hook) PostActStage"       push!(hook, PostActStage(), policy, env)

                if check!(stop_condition, policy, env)
                    is_stop = true
                    break
                end

                if check!(reset_condition, multiagent_policy, env)
                    break
                end
            end
        end # end of an episode

        @timeit_debug timer "push!(policy) PostEpisodeStage"         push!(multiagent_policy, PostEpisodeStage(), env)  # let the policy see the last observation
        @timeit_debug timer "optimise! PostEpisodeStage"             optimise!(multiagent_policy, PostEpisodeStage())
        @timeit_debug timer "push!(hook) PostEpisodeStage"           push!(multiagent_hook, PostEpisodeStage(), multiagent_policy, env)
    end
    push!(multiagent_policy, PostExperimentStage(), env)
    push!(multiagent_hook, PostExperimentStage(), multiagent_policy, env)
    multiagent_policy
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
    stop_condition::AbstractStopCondition,
    hook::MultiAgentHook,
    reset_condition::AbstractResetCondition=ResetIfEnvTerminated(),
) where {E<:AbstractEnv}
    RLCore._run(
        multiagent_policy,
        env,
        stop_condition,
        hook,
        reset_condition,
    )
end

# Default behavior for multi-agent, simultaneous `push!` is to iterate over all players and call `push!` on the appropriate policy
function Base.push!(multiagent::MultiAgentPolicy, stage::S, env::E) where {S<:AbstractStage, E<:AbstractEnv}
    for player in players(env)
        push!(multiagent[player], stage, env, player)
    end
end

# Like in the single-agent case, push! at the PostActStage() calls push! on each player.
function Base.push!(agent::Agent, ::PreEpisodeStage, env::AbstractEnv, player::Player)
    push!(agent.trajectory, (state = state(env, player),))
end

function Base.push!(multiagent::MultiAgentPolicy, s::PreEpisodeStage, env::E) where {E<:AbstractEnv}
    for player in players(env)
        push!(multiagent[player], s, env, player)
    end
end

function RLBase.plan!(agent::Agent, env::AbstractEnv, player::Player)
    RLBase.plan!(agent.policy, env, player)
end

# Like in the single-agent case, push! at the PostActStage() calls push! on each player to store the action, reward, next_state, and terminal signal.
function Base.push!(multiagent::MultiAgentPolicy, ::PostActStage, env::E, actions) where {E<:AbstractEnv}
    for (player, action) in zip(players(env), actions)
        next_state = state(env, player)
        observation = (
            state = next_state,
            action = action,
            reward = reward(env, player),
            terminal = is_terminated(env)
        )
        push!(multiagent[player].trajectory, observation)
    end
end

function Base.push!(agent::Agent, ::PostEpisodeStage, env::AbstractEnv, player::Player)
    if haskey(agent.trajectory, :next_action) 
        action = RLBase.plan!(agent.policy, env, p)
        push!(agent.trajectory, PartialNamedTuple((action = action, )))
    end
end

function Base.push!(hook::MultiAgentHook, stage::S, multiagent::MultiAgentPolicy, env::E) where {E<:AbstractEnv, S<:AbstractStage}
    for player in players(env)
        push!(hook[player], stage, multiagent[player], env, player)
    end
end

@inline function _push!(stage::AbstractStage, policy::P, env::E, player::Player, hook::H, hook_tuple...) where {P <: AbstractPolicy, E <: AbstractEnv, H <: AbstractHook}
    push!(hook, stage, policy, env, player)
    _push!(stage, policy, env, player, hook_tuple...)
end

_push!(stage::AbstractStage, policy::P, env::E, player::Player) where {P <: AbstractPolicy, E <: AbstractEnv} = nothing

function Base.push!(composed_hook::ComposedHook{T},
                            stage::AbstractStage,
                            policy::P,
                            env::E,
                            player::Player
                            ) where {T <: Tuple, P <: AbstractPolicy, E <: AbstractEnv}
    _push!(stage, policy, env, player, composed_hook.hooks...)
end

#For simultaneous players, plan! returns a Tuple of actions. 
function RLBase.plan!(multiagent::MultiAgentPolicy, env::E) where {E<:AbstractEnv}
    return Tuple(RLBase.plan!(multiagent[player], env, player) for player in players(env))
end

function RLBase.optimise!(multiagent::MultiAgentPolicy, stage::S) where {S<:AbstractStage}
    for policy in multiagent
        RLCore.optimise!(policy, stage)
    end
end
