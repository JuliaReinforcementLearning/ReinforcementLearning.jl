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
