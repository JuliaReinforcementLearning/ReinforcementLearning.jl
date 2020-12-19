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

for f in vcat(RLBase.ENV_API, RLBase.MULTI_AGENT_ENV_API)
    if f ∉ (:DefaultStateStyle, )
        @eval RLBase.$f(x::DefaultStateStyleEnv, args...; kwargs...) =
            $f(x.env, args...; kwargs...)
    end
end

(env::DefaultStateStyleEnv)(args...; kwargs...) = env.env(args...; kwargs...)

RLBase.state(env::DefaultStateStyleEnv, ss::RLBase.AbstractStateStyle) = state(env.env, ss)
RLBase.state_space(env::DefaultStateStyleEnv, ss::RLBase.AbstractStateStyle) =
    state_space(env.env, ss)
