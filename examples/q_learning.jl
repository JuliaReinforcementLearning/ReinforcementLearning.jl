using Flux
using ReinforcementLearningEnvironments
using ReinforcementLearning

env = CartPoleEnv()
ns, na = length(observation_space(env)), length(action_space(env))

function loss_cal(ŷ, y)
    (ŷ .- y) .^ 2
end

agent = Agent(
    QBasedPolicy(
        QLearner(
            approximator = NeuralNetworkQ(
                Chain(Dense(ns, 128, relu), Dense(128, 128, relu), Dense(128, na)),
                ADAM(0.0005),
            ),
            loss_fun = loss_cal,
        ),
        EpsilonGreedySelector{:exp}(ϵ_stable = 0.01, decay_steps = 500),
    ),
    circular_RTSA_buffer(
        capacity = 10000,
        state_eltype = Vector{Float64},
        state_size = (ns,),
    ),
)

hook = TotalRewardPerEpisode()

run(agent, env, StopAfterStep(10000); hook = hook)