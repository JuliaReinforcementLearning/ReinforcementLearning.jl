export SequentialEnv

"""
    SequentialEnv(env)

Turn a simultaneous `env` into a sequential env.
"""
mutable struct SequentialEnv{E<:AbstractEnv} <: AbstractEnvWrapper
    env::E
    current_player_idx::Int
    actions::Vector{Any}
    function SequentialEnv(env::T) where {T<:AbstractEnv}
        @assert DynamicStyle(env) === SIMULTANEOUS "The SequentialEnv wrapper can only be applied to SIMULTANEOUS environments"
        new{T}(env, 1, Vector{Any}(undef, length(players(env))))
    end
end

RLBase.DynamicStyle(env::SequentialEnv) = SEQUENTIAL

RLBase.current_player(env::SequentialEnv) = env.current_player_idx

RLBase.legal_action_space(env::SequentialEnv) = legal_action_space(env, current_player(env))
RLBase.legal_action_space(env::SequentialEnv, player) = legal_action_space(env.env, player)

RLBase.action_space(env::SequentialEnv) = action_space(env, current_player(env))
RLBase.action_space(env::SequentialEnv, player) = action_space(env.env, player)

function RLBase.reset!(env::SequentialEnv)
    env.current_player_idx = 1
    reset!(env.env)
end

RLBase.reward(env::SequentialEnv) = reward(env, current_player(env))

RLBase.reward(env::SequentialEnv, player) =
    current_player(env) == 1 ? reward(env.env, player) : 0

function (env::SequentialEnv)(action)
    env.actions[env.current_player_idx] = action
    if env.current_player_idx == length(env.actions)
        env.env(env.actions)
        env.current_player_idx = 1
    else
        env.current_player_idx += 1
    end
end
