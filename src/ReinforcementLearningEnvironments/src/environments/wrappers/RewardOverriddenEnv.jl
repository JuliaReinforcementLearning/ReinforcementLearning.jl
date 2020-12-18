export RewardOverriddenEnv

struct RewardOverriddenEnv{F,E<:AbstractEnv} <: AbstractEnvWrapper
    env::E
    f::F
end

(env::RewardOverriddenEnv)(args...; kwargs...) = env.env(args...; kwargs...)

RewardOverriddenEnv(f) = env -> RewardOverriddenEnv(f, env)

for f in vcat(RLBase.ENV_API, RLBase.MULTI_AGENT_ENV_API)
    if f != :reward
        @eval RLBase.$f(x::RewardOverriddenEnv, args...; kwargs...) =
            $f(x.env, args...; kwargs...)
    end
end

RLBase.reward(env::RewardOverriddenEnv, args...; kwargs...) =
    env.f(reward(env.env, args...; kwargs...))

RLBase.state(env::RewardOverriddenEnv, ss::RLBase.AbstractStateStyle) = state(env.env, ss)
RLBase.state_space(env::RewardOverriddenEnv, ss::RLBase.AbstractStateStyle) =
    state_space(env.env, ss)
