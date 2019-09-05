export AbstractLearner

abstract type AbstractLearner end

approximator(x::AbstractLearner) = x.approximator

(learner::AbstractLearner)(obs) = approximator(learner)(get_state(obs))
(learner::AbstractLearner)(obs, a) = approximator(learner)(get_state(obs), a)