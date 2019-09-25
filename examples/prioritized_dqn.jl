using Flux
using ReinforcementLearningEnvironments
using ReinforcementLearning

env = CartPoleEnv()
ns, na = length(observation_space(env)), length(action_space(env))

model = Chain(Dense(ns, 128, relu), Dense(128, 128, relu), Dense(128, na))

target_model = Chain(Dense(ns, 128, relu), Dense(128, 128, relu), Dense(128, na))

Q = NeuralNetworkQ(model, ADAM(0.0005))
Qₜ = NeuralNetworkQ(target_model, ADAM(0.0005))

function loss_cal(ŷ, y)
    (ŷ .- y) .^ 2
end

agent = Agent(
    QBasedPolicy(
        PrioritizedDQNLearner(
            approximator = Q,
            target_approximator = Qₜ,
            loss_fun = loss_cal,
        ),
        EpsilonGreedySelector{:exp}(ϵ_stable = 0.01, decay_steps = 500),
    ),
    circular_PRTSA_buffer(
        capacity = 10000,
        state_eltype = Vector{Float64},
        state_size = (ns,),
    ),
)

hook = TotalRewardPerEpisode()

run(agent, env, StopAfterStep(10000); hook = hook)