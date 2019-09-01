export AbstractLearner

abstract type AbstractLearner end

approximator(x::AbstractLearner) = x.approximator

(learner::AbstractLearner)(s) = approximator(learner)(s)
(learner::AbstractLearner)(s, a) = approximator(learner)(s, a)