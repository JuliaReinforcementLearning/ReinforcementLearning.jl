export StateCachedEnv

"""
Cache the state so that `state(env)` will always return the same result before
the next interaction with `env`. This function is useful because some
environments are stateful during each `state(env)`. For example:
`StateTransformedEnv(StackFrames(...))`.
"""
mutable struct StateCachedEnv{S,E <: AbstractEnv} <: AbstractEnvWrapper
    s::S
    env::E
    is_state_cached::Bool
end

StateCachedEnv(env) = StateCachedEnv(state(env), env, true)

function (env::StateCachedEnv)(args...; kwargs...)
    env.env(args...; kwargs...)
    env.is_state_cached = false
end

function RLBase.state(env::StateCachedEnv, args...; kwargs...)
    if env.is_state_cached
        env.s
    else
        env.s = state(env.env, args...; kwargs...)
        env.is_state_cached = true
        env.s
    end
end
