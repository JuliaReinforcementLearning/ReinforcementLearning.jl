using Flux
using ReinforcementLearningEnvironments
using ReinforcementLearning

env = CartPoleEnv()
ns, na = length(observation_space(env)), length(action_space(env))
model = Chain(
    Dense(ns, 128, relu),
    Dense(128, 128, relu),
    Dense(128, na)
)

function loss_cal(ŷ, y)
    (ŷ .- y).^2
end

app = NeuralNetworkQ(model, ADAM(0.0005))
learner = QLearner(app, loss_cal;γ=0.99f0)
buffer =  circular_RTSA_buffer(;capacity=10000, state_eltype=Vector{Float64}, state_size=(ns,))
selector = EpsilonGreedySelector(0.01;decay_steps=500, decay_method=:exp)
agent = DQN(learner, buffer, selector)

hook=TotalRewardPerEpisode()

train(agent, env, StopAfterStep(10000);hook=hook)