export CloneStatePreprocessor

#####
# CloneStatePreprocessor
#####

struct CloneStatePreprocessor <: AbstractPreprocessor end

(p::CloneStatePreprocessor)(obs) = StateOverriddenObs(obs, deepcopy(get_state(obs)))
