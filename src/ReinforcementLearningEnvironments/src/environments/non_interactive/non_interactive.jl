export NonInteractiveEnv

abstract type NonInteractiveEnv <: AbstractEnv end 
(env::NonInteractiveEnv)() = env(nothing)

include("pendulum.jl")
