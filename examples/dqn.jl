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

app = NeuralNetworkQ(model, ADAM(0.0005))
learner = QLearner(app, Flux.mse;γ=0.99)
buffer =  circular_RTSA_buffer(;capacity=10000, state_eltype=Vector{Float64}, state_size=(ns,))
selector = EpsilonGreedySelector(0.01;decay_steps=500, decay_method=:exp)
agent = DQN(learner, buffer, selector;γ=0.99)

function f(n)
    i = 0
    (x...) -> begin
       res = i >= n
       i += 1
       res
    end
end

rewards = []
losses = []

function f_pre_episode_hook(agent, env, obs)
    push!(rewards, [])
    push!(losses, [])
end

function f_post_act_hook(agent, env, obs, action)
    push!(rewards[end], obs.reward)
    push!(losses[end], agent.learner.loss)
end

train(agent, env, f(10000); pre_episode_hook=f_pre_episode_hook, post_act_hook=f_post_act_hook)