export NonInteractiveEnv

abstract type NonInteractiveEnv <: AbstractEnv end
(env::NonInteractiveEnv)() = env(nothing)
RLBase.get_actions(::NonInteractiveEnv) = EmptySpace()

include("pendulum.jl")
