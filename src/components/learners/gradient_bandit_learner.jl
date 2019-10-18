export GradientBanditLearner

using Flux: softmax, onehot

"""
    GradientBanditLearner(;approximator::A, optimizer::O, baseline::B)
"""
Base.@kwdef mutable struct GradientBanditLearner{A,O,B} <: AbstractLearner
    approximator::A
    optimizer::O
    baseline::B
end

(learner::GradientBanditLearner)(obs::Observation) =
    obs |> get_state |> learner.approximator |> softmax

function update!(learner::GradientBanditLearner, transitions)
    s, a, r = transitions
    probs = s |> learner.approximator |> softmax
    r̄ = learner.baseline isa Number ? learner.baseline : learner.baseline(r)
    errors = (r - r̄) .* (onehot(a, 1:length(probs)) .- probs)
    update!(learner.approximator, s => apply!(learner.optimizer, s, errors))
end

function extract_transitions(buffer::EpisodeTurnBuffer, ::GradientBanditLearner)
    if length(buffer) > 0
        state(buffer)[end-1], action(buffer)[end-1], reward(buffer)[end]
    else
        nothing
    end
end
