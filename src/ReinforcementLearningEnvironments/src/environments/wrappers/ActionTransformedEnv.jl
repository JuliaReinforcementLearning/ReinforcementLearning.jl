export ActionTransformedEnv

struct ActionTransformedEnv{P,M,E<:AbstractEnv} <: AbstractEnvWrapper
    env::E
    action_mapping::M
    action_space_mapping::P
end

"""
    ActionTransformedEnv(env;action_space_mapping=identity, action_mapping=identity)

`action_space_mapping` will be applied to `action_space(env)` and
`legal_action_space(env)`. `action_mapping` will be applied to `action` before
feeding it into `env`.
"""
ActionTransformedEnv(env; action_mapping = identity, action_space_mapping = identity) = 
    ActionTransformedEnv(env, action_mapping, action_space_mapping)

Base.copy(env::ActionTransformedEnv) = 
    ActionTransformedEnv(
        copy(env.env), 
        action_mapping = env.action_mapping, 
        action_space_mapping = env.action_space_mapping
    )

RLBase.action_space(env::ActionTransformedEnv, args...) =
    env.action_space_mapping(action_space(env.env, args...))

RLBase.legal_action_space(env::ActionTransformedEnv, args...) =
    env.action_space_mapping(legal_action_space(env.env, args...))

(env::ActionTransformedEnv)(action, args...; kwargs...) =
    env.env(env.action_mapping(action), args...; kwargs...)
