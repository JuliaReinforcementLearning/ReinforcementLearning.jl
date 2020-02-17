export CloneStatePreprocessor, ComposedPreprocessor

(p::AbstractPreprocessor)(obs) = StateOverriddenObs(obs = obs, state = p(get_state(obs)))

struct ComposedPreprocessor{T} <: AbstractPreprocessor
    preprocessors::T
end

ComposedPreprocessor(p::AbstractPreprocessor...) = ComposedPreprocessor(p)
(p::ComposedPreprocessor)(obs) = reduce((x, f) -> f(x), p.preprocessors, init = obs)

#####
# CloneStatePreprocessor
#####

struct CloneStatePreprocessor <: AbstractPreprocessor end

(p::CloneStatePreprocessor)(obs) = StateOverriddenObs(obs, deepcopy(get_state(obs)))
