export MaxTimeoutEnv

mutable struct MaxTimeoutEnv{E<:AbstractEnv} <: AbstractEnvWrapper
    env::E
    max_t::Int
    current_t::Int
end

"""
    MaxTimeoutEnv(env::E, max_t::Int; current_t::Int = 1)

Force `is_terminated(env)` return `true` after `max_t` interactions.
"""
MaxTimeoutEnv(env::E, max_t::Int; current_t::Int = 1) where {E<:AbstractEnv} =
    MaxTimeoutEnv(env, max_t, current_t)

function (env::MaxTimeoutEnv)(args...; kwargs...)
    env.env(args...; kwargs...)
    env.current_t = env.current_t + 1
end

for f in vcat(RLBase.ENV_API, RLBase.MULTI_AGENT_ENV_API)
    if f != :terminal
        @eval RLBase.$f(x::MaxTimeoutEnv, args...; kwargs...) =
            $f(x.env, args...; kwargs...)
    end
end

RLBase.is_terminated(env::MaxTimeoutEnv) =
    (env.current_t > env.max_t) || is_terminated(env.env)


RLBase.state(env::MaxTimeoutEnv, ss::RLBase.AbstractStateStyle) = state(env.env, ss)
RLBase.state_space(env::MaxTimeoutEnv, ss::RLBase.AbstractStateStyle) =
    state_space(env.env, ss)
