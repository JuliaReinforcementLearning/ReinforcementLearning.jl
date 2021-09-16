export DefaultStateStyleEnv

struct DefaultStateStyleEnv{S,E} <: AbstractEnvWrapper
    env::E
end

"""
    DefaultStateStyleEnv{S}(env::E)

Reset the result of `DefaultStateStyle` without changing the original behavior.
"""
DefaultStateStyleEnv{S}(env::E) where {S,E} = DefaultStateStyleEnv{S,E}(env)

RLBase.DefaultStateStyle(::DefaultStateStyleEnv{S}) where {S} = S

Base.copy(env::DefaultStateStyleEnv{S}) where S = DefaultStateStyleEnv{S}(copy(env.env))

RLBase.state(env::DefaultStateStyleEnv{S}) where S = state(env.env, S)
RLBase.state(env::DefaultStateStyleEnv, ss::RLBase.AbstractStateStyle) = state(env.env, ss)
RLBase.state(env::DefaultStateStyleEnv{S}, player) where S = state(env.env, S, player)

RLBase.state_space(env::DefaultStateStyleEnv{S}) where S = state_space(env.env, S)
RLBase.state_space(env::DefaultStateStyleEnv, ss::RLBase.AbstractStateStyle) = state_space(env.env, ss)