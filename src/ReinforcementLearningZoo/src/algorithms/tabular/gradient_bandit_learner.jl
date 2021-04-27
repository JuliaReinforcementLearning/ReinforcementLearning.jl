export GradientBanditLearner

using Flux: softmax, onehot

Base.@kwdef struct GradientBanditLearner{A,B} <: AbstractLearner
    approximator::A
    baseline::B
end

(learner::GradientBanditLearner)(s::Int) = s |> learner.approximator |> softmax
(learner::GradientBanditLearner)(env::AbstractEnv) = learner(state(env))

function RLBase.update!(
    L::GradientBanditLearner,
    t::AbstractTrajectory,
    ::AbstractEnv,
    ::PreActStage,
) end

function RLBase.update!(
    L::GradientBanditLearner,
    t::AbstractTrajectory,
    ::AbstractEnv,
    ::PostActStage,
)
    A = L.approximator
    s, a, r = t[:state][end], t[:action][end], t[:reward][end]
    probs = s |> A |> softmax
    r̄ = L.baseline isa Number ? L.baseline : L.baseline(r)
    errors = (r - r̄) .* (onehot(a, 1:length(probs)) .- probs)
    update!(A, s => -errors)
end

function RLBase.update!(
    t::AbstractTrajectory,
    ::QBasedPolicy{<:GradientBanditLearner},
    ::AbstractEnv,
    ::PreEpisodeStage,
)
    empty!(t)
end
