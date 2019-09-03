export GradientBanditLearner

using Flux:softmax, onehot

mutable struct GradientBanditLearner{A, O, B} <: AbstractLearner
    approximator::A
    optimizer::O
    baseline::B
end

GradientBanditLearner(;approximator, optimizer, baseline) = GradientBanditLearner(approximator, optimizer, baseline)

(learner::GradientBanditLearner)(s) = s |> learner.approximator |> softmax

function update!(learner::GradientBanditLearner, s, a, r)
    probs = learner(s)
    r̄ = learner.baseline isa Number ? learner.baseline : learner.baseline(r)
    errors = (r - r̄) .* (onehot(a, 1:length(probs)) .- probs)
    update!(learner.approximator, s => apply!(learner.optimizer, s, errors))
end