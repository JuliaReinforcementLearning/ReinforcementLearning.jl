export SequentialEnv

"""
    SequentialEnv(env)

Turn a simultaneous `env` into a sequential env.
"""
mutable struct SequentialEnv{E<:AbstractEnv} <: AbstractEnvWrapper
    env::E
    current_player_idx::Int
    actions::Vector{Any}
    function SequentialEnv(env::T) where T<:AbstractEnv
        @assert DynamicStyle(env) === SIMULTANEOUS "The SequentialEnv wrapper can only be applied to SIMULTANEOUS environments"
        new{T}(env, 1, Vector{Any}(undef, length(players(env))))
    end
end

RLBase.DynamicStyle(env::SequentialEnv) = SEQUENTIAL

RLBase.current_player(env::SequentialEnv) = env.current_player_idx

RLBase.action_space(env::SequentialEnv) = action_space(env, current_player(env))
RLBase.action_space(env::SequentialEnv, player) = action_space(env.env, player)

function (env::SequentialEnv)(action)
    env.actions[env.current_player_idx] = action
    if env.current_player_idx == length(env.actions)
        env.env(env.actions)
        env.current_player_idx = 1
    else
        env.current_player_idx += 1
    end
end

