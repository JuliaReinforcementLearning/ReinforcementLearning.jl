using ReinforcementLearning
using StableRNGs
using Flux
using IntervalSets

function cartpole_reward_function(env; action=env.last_action, nstate=state(env))
    arm_length = 2 * env.params.halflength
    x, xvel, theta, thetavel = nstate
    reward = exp(-(x / arm_length - sin(theta))^2 - (1 + cos(theta))^2)
    reward -= 0.01 * sum(action^2)
    return reward
end

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:PETS},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    rng = StableRNG(seed)
    inner_env = CartPoleEnv(T = Float32, continuous = true, rng = rng)

    A = action_space(inner_env)
    low = A.left
    high = A.right

    env = ActionTransformedEnv(
        StateTransformedEnv(
            RewardTransformedEnv(
                inner_env;
                reward_mapping = cartpole_reward_function,
            );
            state_mapping = x -> [x[1:2]; sin(x[3]); cos(x[3]); x[4]], # TODO: this does not seem very efficient?
        );
        action_mapping = x -> low + (x[1] + 1) * 0.5 * (high - low), 
    )

    ns = length(state(env))
    na = 1

    init = glorot_uniform(rng)

    agent = Agent(
        policy = PETSPolicy(
            optimizer = CEMTrajectoryOptimizer(
                lower_bound = [low],
                upper_bound = [high],
                population = 500,
                elite_ratio = 0.1,
                iterations = 5,
                horizon = 15,
                α = 0.1,
                rng = rng,
            ),
            ensamble = [
                NeuralNetworkApproximator(
                    model = GaussianNetwork(
                        pre = Chain(
                            Dense(ns + na, 30, relu, init = init), 
                            Dense(30, 30, relu, init = init),
                        ),
                        μ = Chain(Dense(30, ns + 1, init = init)),
                        logσ = Chain(
                            Dense(30, ns + 1, init = init),
                        ),
                        clampfun = softclamp,
                        min_σ = 1f-5,
                        max_σ = 1f2, 
                    ),
                    optimizer = ADAM(7.5e-4),
                ) for _ in 1:5
            ],
            batch_size = 256,
            start_steps = 200,
            start_policy = RandomPolicy(Space([-1.0..1.0 for _ in 1:na]); rng = rng),
            update_after = 200,
            update_freq = 50,
            rng = rng,
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 5000,
            state = Vector{Float32} => (ns,),
            action = Vector{Float32} => (na,),
        ),
    )

    stop_condition = StopAfterStep(5000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook, "# Play CartPole with PETS")
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_PETS_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_PETS_CartPole.png") #hide

# ![](assets/JuliaRL_PETS_CartPole.png)