export AbstractLearner

"""
A learner is used to define:

- How to generate necessary training data?
- How to update the inner [Approximators](@ref)?
"""
abstract type AbstractLearner end

approximator(x::AbstractLearner) = x.approximator

(learner::AbstractLearner)(obs::Observation) = approximator(learner)(get_state(obs))
(learner::AbstractLearner)(obs::Observation, a) = approximator(learner)(get_state(obs), a)
(learner::AbstractLearner)(s) = approximator(learner)(s)
(learner::AbstractLearner)(s, a) = approximator(learner)(s, a)