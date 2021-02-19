export StateOverriddenEnv

"""
    StateOverriddenEnv(f, env)

Apply `f` to override `state(env)`.

!!! note
    If the meaning of state space is changed after apply `f`, one should
    manually redefine the `RLBase.state_space(env::YourSpecificEnv)`.
"""
struct StateOverriddenEnv{F,E <: AbstractEnv} <: AbstractEnvWrapper
    env::E
    f::F
end

StateOverriddenEnv(f) = env -> StateOverriddenEnv(f, env)

RLBase.state(env::StateOverriddenEnv, args...; kwargs...) =
    env.f(state(env.env, args...; kwargs...))
