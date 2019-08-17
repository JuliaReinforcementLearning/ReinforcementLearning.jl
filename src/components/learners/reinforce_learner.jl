export ReinforceLearner, update!

struct ReinforceLearner{Tapp<:QApproximator} <: AbstractLearner{Tapp}
    approximator::Tapp
    α::Float64
    γ::Float64
    loss::Float32
end

function update!(learner::ReinforceLearner, states, actions, rewards)
    π, α, γ = learner.approximator, learner.α, learner.γ

    gains = discount_rewards(rewards)
    loss = sum(i -> - π(states[i], actions[i]) * gains[i], 1:length(gains))
    learner.loss = loss.data
    update!(π, loss)
end