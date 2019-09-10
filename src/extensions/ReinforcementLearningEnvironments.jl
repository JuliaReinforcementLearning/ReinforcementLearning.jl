import ReinforcementLearningEnvironments: get_terminal

get_terminal(obs::Vector{<:Observation}) = all(get_terminal(o) for o in obs)

reset(obs::Observation; reward=get_reward(obs), terminal=get_terminal(obs), state=get_state(obs), meta=obs.meta) = Observation(reward, terminal, state, meta)