export AbstractEnvWrapper

abstract type AbstractEnvWrapper <: AbstractEnv end

Base.nameof(env::AbstractEnvWrapper) = "$(nameof(env.env)) |> $(nameof(typeof(env)))"

Base.getindex(env::AbstractEnvWrapper) = env.env

(env::AbstractEnvWrapper)(args...; kwargs...) = env.env(args...; kwargs...)

for f in vcat(RLBase.ENV_API, RLBase.MULTI_AGENT_ENV_API)
    @eval RLBase.$f(x::AbstractEnvWrapper, args...; kwargs...) = $f(x[], args...; kwargs...)
end

# avoid ambiguous
RLBase.state(env::AbstractEnvWrapper, ss::RLBase.AbstractStateStyle, p) =
    state(env[], ss, p)
RLBase.state(env::AbstractEnvWrapper, ss::RLBase.AbstractStateStyle) = state(env[], ss)
RLBase.state_space(env::AbstractEnvWrapper, ss::RLBase.AbstractStateStyle) =
    state_space(env[], ss)
RLBase.state_space(env::AbstractEnvWrapper, ss::RLBase.AbstractStateStyle, p) =
    state_space(env[], ss, p)

include("ActionTransformedEnv.jl")
include("DefaultStateStyle.jl")
include("MaxTimeoutEnv.jl")
include("RewardOverriddenEnv.jl")
include("StateCachedEnv.jl")
include("StateOverriddenEnv.jl")
include("StochasticEnv.jl")
