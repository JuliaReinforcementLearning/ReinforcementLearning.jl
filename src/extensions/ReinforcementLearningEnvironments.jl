import ReinforcementLearningEnvironments: get_terminal

get_terminal(obs::Vector{<:Observation}) = all(get_terminal(o) for o in obs)