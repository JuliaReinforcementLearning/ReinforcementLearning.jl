export WrappedEnv

import ReinforcementLearningEnvironments: get_terminal,
                                          observation_space,
                                          action_space,
                                          observe,
                                          interact!,
                                          reset!

get_terminal(obs::Vector{<:Observation}) = all(get_terminal(o) for o in obs)

reset(
    obs::Observation;
    reward = get_reward(obs),
    terminal = get_terminal(obs),
    state = get_state(obs),
    meta = obs.meta,
) = Observation(reward, terminal, state, meta)

#####
# WrappedEnv
#####

"""
    WrappedEnv(;env, preprocessor)

The observation of `env` is first processed by the `preprocessor`.
"""
Base.@kwdef struct WrappedEnv{E<:AbstractEnv,P} <: AbstractEnv
    env::E
    preprocessor::P
end

observation_space(env::WrappedEnv) = observation_space(env.env)
action_space(env::WrappedEnv) = action_space(env.env)
interact!(env::WrappedEnv, a) = interact!(env.env, a)
observe(env::WrappedEnv, args...) = env.env |> observe |> env.preprocessor

reset!(env::WrappedEnv) = reset!(env.env)