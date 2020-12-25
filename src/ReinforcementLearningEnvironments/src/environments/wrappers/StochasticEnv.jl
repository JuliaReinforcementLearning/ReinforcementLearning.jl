export StochasticEnv

using StatsBase: sample, Weights

struct StochasticEnv{E<:AbstractEnv,R} <: AbstractEnv
    env::E
    rng::R
end

function StochasticEnv(env; rng = Random.GLOBAL_RNG)
    ChanceStyle(env) === EXPLICIT_STOCHASTIC ||
        throw(ArgumentError("only environments of EXPLICIT_STOCHASTIC style is supported"))
    env = StochasticEnv(env, rng)
    reset!(env)
    env
end

function RLBase.reset!(env::StochasticEnv)
    reset!(env.env)
    while current_player(env.env) == chance_player(env.env)
        p = prob(env.env)
        A = action_space(env.env)
        x = A[sample(env.rng, Weights(p, 1.0))]
        env.env(x)
    end
end

function (env::StochasticEnv)(a)
    env.env(a)
    while current_player(env.env) == chance_player(env.env)
        p = prob(env.env)
        A = action_space(env.env)
        x = A[sample(env.rng, Weights(p, 1.0))]
        env.env(x)
    end
end

RLBase.ChanceStyle(::StochasticEnv) = STOCHASTIC
RLBase.players(env::StochasticEnv) =
    [p for p in players(env.env) if p != chance_player(env.env)]
Random.seed!(env::StochasticEnv, s) = Random.seed!(env.rng, s)

for f in vcat(RLBase.ENV_API, RLBase.MULTI_AGENT_ENV_API)
    if f ∉ (:players, :ChanceStyle, :reset!)
        @eval RLBase.$f(x::StochasticEnv, args...; kwargs...) =
            $f(x.env, args...; kwargs...)
    end
end

RLBase.state(env::StochasticEnv, ss::RLBase.AbstractStateStyle) = state(env.env, ss)
RLBase.state_space(env::StochasticEnv, ss::RLBase.AbstractStateStyle) =
    state_space(env.env, ss)
