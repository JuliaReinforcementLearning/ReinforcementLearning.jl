export StateTransformedEnv

struct StateTransformedEnv{P,M,E<:AbstractEnv} <: AbstractEnvWrapper
    env::E
    state_mapping::P
    state_space_mapping::M
end

"""
    StateTransformedEnv(env; state_mapping=identity, state_space_mapping=identity)

`state_mapping` will be applied on the original state when calling `state(env)`,
and similarly `state_space_mapping` will be applied when calling `state_space(env)`.
"""
StateTransformedEnv(env; state_mapping = identity, state_space_mapping = identity) =
    StateTransformedEnv(env, state_mapping, state_space_mapping)

RLBase.state(env::StateTransformedEnv, args...; kwargs...) =
    env.state_mapping(state(env.env, args...; kwargs...))

RLBase.state_space(env::StateTransformedEnv, args...; kwargs...) =
    env.state_space_mapping(state_space(env.env, args...; kwargs...))
