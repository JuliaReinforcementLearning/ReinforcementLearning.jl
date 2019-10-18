export WrappedEnv

import ReinforcementLearningEnvironments: get_terminal,
                                          observation_space,
                                          action_space,
                                          observe,
                                          interact!,
                                          reset!

@doc """
    Observation(;reward, terminal, state, meta...)

The observation of an environment from the perspective of an agent.

# Keywords & Fields

- `reward`: the reward of an agent
- `terminal`: indicates that if the environment is terminated or not.
- `state`: the current state of the environment from the perspective of an agent
- `meta`: some other information, like `legal_actions`...

!!! note
    The `reward` and `terminal` of the first observation before interacting with an environment may not be valid.
"""
Observation

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
reset!(env::WrappedEnv) = reset!(env.env)
observe(env::WrappedEnv, args...) = env.env |> observe |> env.preprocessor