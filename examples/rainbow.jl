using Flux
using ReinforcementLearningEnvironments
using ReinforcementLearning

n_atoms = 51

env = CartPoleEnv()
ns, na = length(observation_space(env)), length(action_space(env))
model = Chain(
    Dense(ns, 128, relu),
    Dense(128, 128, relu),
    Dense(128, na*n_atoms)
)

target_model = Chain(
    Dense(ns, 128, relu),
    Dense(128, 128, relu),
    Dense(128, na*n_atoms)
)

Q = NeuralNetworkQ(model, ADAM(0.0005))
Qₜ = NeuralNetworkQ(target_model, ADAM(0.0005))

function logitcrossentropy_expand(logŷ::AbstractVecOrMat, y::AbstractVecOrMat)
  return vec(-sum(y .* logsoftmax(logŷ), dims=1))
end

learner = RainbowLearner(Q, Qₜ, logitcrossentropy_expand;γ=0.99f0, Vₘₐₓ=200.0f0, Vₘᵢₙ=0.0f0, n_actions=na, n_atoms=n_atoms, target_update_freq=100)
buffer =  circular_RTSA_buffer(;capacity=10000, state_eltype=Vector{Float64}, state_size=(ns,))
selector = EpsilonGreedySelector(0.01;decay_steps=500, decay_method=:exp)
agent = DQN(learner, buffer, selector)

hook=TotalRewardPerEpisode()

train(agent, env, StopAfterStep(10000);hook=hook)

