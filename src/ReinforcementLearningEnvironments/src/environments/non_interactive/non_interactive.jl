export NonInteractiveEnv

abstract type NonInteractiveEnv <: AbstractEnv end
(env::NonInteractiveEnv)() = RLBase.act!(env, nothing)
RLBase.action_space(::NonInteractiveEnv) = (nothing,)

include("pendulum.jl")
