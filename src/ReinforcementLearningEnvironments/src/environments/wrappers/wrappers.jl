export AbstractEnvWrapper

abstract type AbstractEnvWrapper <: AbstractEnv end

Base.nameof(env::AbstractEnvWrapper) = "$(nameof(env.env)) |> $(nameof(typeof(env)))"

# wrapped_env[] will get the next layer, be it wrapper or env
Base.getindex(env::AbstractEnvWrapper) = env.env

# wrapped_env[!] will remove all wrapper layers and get the env inside them
Base.getindex(env::AbstractEnvWrapper, ::typeof(!)) = env[][!]
Base.getindex(env::AbstractEnv, ::typeof(!)) = env

(env::AbstractEnvWrapper)(args...; kwargs...) = env[](args...; kwargs...)

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
include("RewardTransformedEnv.jl")
include("StateCachedEnv.jl")
include("StateTransformedEnv.jl")
include("StochasticEnv.jl")
include("SequentialEnv.jl")
