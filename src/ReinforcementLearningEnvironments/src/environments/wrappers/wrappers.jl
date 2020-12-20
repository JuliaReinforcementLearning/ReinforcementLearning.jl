abstract type AbstractEnvWrapper <: AbstractEnv end

Base.nameof(env::AbstractEnvWrapper) = "$(nameof(env.env)) |> $(nameof(typeof(env)))"

include("ActionTransformedEnv.jl")
include("DefaultStateStyle.jl")
include("MaxTimeoutEnv.jl")
include("RewardOverriddenEnv.jl")
include("StateCachedEnv.jl")
include("StateOverriddenEnv.jl")
include("StochasticEnv.jl")
