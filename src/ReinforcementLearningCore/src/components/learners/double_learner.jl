export DoubleLearner

using Random

Base.@kwdef struct DoubleLearner{T1<:AbstractLearner,T2<:AbstractLearner,R<:AbstractRNG} <:
                   AbstractLearner
    L1::T1
    L2::T2
    rng::R = MersenneTwister()
end

DoubleLearner(l1, l2; seed = nothing) = DoubleLearner(l1, l2, MersenneTwister(seed))

(learner::DoubleLearner)(obs) = learner.L1(obs) .+ learner.L2(obs)

RLBase.extract_experience(t::AbstractTrajectory, learner::DoubleLearner) =
    extract_experience(t, learner.L1)

update!(learner::DoubleLearner, experience) =
    rand(learner.rng, Bool) ? update!(learner.L1, experience) :
    update!(learner.L2, experience)
