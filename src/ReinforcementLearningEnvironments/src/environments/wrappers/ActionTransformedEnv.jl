export ActionTransformedEnv

struct ActionTransformedEnv{P,M,E<:AbstractEnv} <: AbstractEnvWrapper
    action_space_mapping::P
    action_mapping::M
    env::E
end

"""
    ActionTransformedEnv(env;action_space_mapping=identity, action_mapping=identity)

`action_space_mapping` will be applied to `action_space(env)` and
`legal_action_space(env)`. `action_mapping` will be applied to `action` before
feeding it into `env`.
"""
function ActionTransformedEnv(
    env;
    action_space_mapping = identity,
    action_mapping = identity,
)
    ActionTransformedEnv(action_space_mapping, action_mapping, env)
end

RLBase.action_space(env::ActionTransformedEnv, args...) =
    env.action_space_mapping(action_space(env.env), args...)

RLBase.legal_action_space(env::ActionTransformedEnv, args...) =
    env.action_space_mapping(legal_action_space(env.env), args...)

(env::ActionTransformedEnv)(action, args...; kwargs...) =
    env.env(env.action_mapping(action), args...; kwargs...)
