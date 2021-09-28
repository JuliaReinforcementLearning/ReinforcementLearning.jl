using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using IntervalSets
using Distributions

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
    ns = length(state(inner_env))
    na = 1

    ensamble_size = 5

    env = ActionTransformedEnv(
        inner_env;
        action_mapping = x -> low + (x[1] + 1) * 0.5 * (high - low),
    )
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
                return_mean_elites = true,
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
                    optimizer = ADAM(0.003),
                ) for i in 1:ensamble_size
            ],
            batch_size = 64,
            start_steps = 100,
            start_policy = RandomPolicy(Space([-1.0..1.0 for _ in 1:na]); rng = rng),
            update_after = 100,
            update_freq = 100,
            rng = rng,
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 10000,
            state = Vector{Float32} => (ns,),
            action = Vector{Float32} => (na,),
        ),
    )

    stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
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