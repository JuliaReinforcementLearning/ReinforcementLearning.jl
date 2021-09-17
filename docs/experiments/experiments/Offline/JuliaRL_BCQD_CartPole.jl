# ---
# title: JuliaRL\_BCQD\_CartPole
# cover: assets/JuliaRL_BCQD_CartPole_medium.png
# description: BCQD applied to CartPole
# date: 2021-09-17
# author: "[Guoyu Yang](https://github.com/pilgrimygy)"
# ---

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BCQD},
    ::Val{:CartPole},
    type::AbstractString;
    save_dir = nothing,
    seed = 123,
)
    rng = StableRNG(seed)
    UPDATE_FREQ = 1
    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))

    trajectory_num = 10000
    dataset_size = 10000
    batch_size = 64

    init = glorot_uniform(rng)

    create_actor_critic() = ActorCritic(
        actor = Chain(
            Dense(ns, 128, relu; init),
            Dense(128, 128, relu; init),
            Dense(128, na; init),
        ),
        critic = Chain(
            Dense(ns, 128, relu; init),
            Dense(128, 128, relu; init),
            Dense(128, na; init),
        ),
        optimizer = ADAM(1e-3),
    )

    agent = Agent(
        policy = OfflinePolicy(
            learner = BCQDLearner(
                approximator = create_actor_critic() |> cpu,
                target_approximator = create_actor_critic() |> cpu,
                γ = 0.99f0,
                τ = 0.01f0,
                θ = 1f-2,
                threshold = 0.3f0,
                batch_size = batch_size,
                update_freq = UPDATE_FREQ,
            ),
            dataset = gen_JuliaRL_dataset(:BasicDQN, :CartPole, type; dataset_size = dataset_size),
            continuous = false,
            batch_size = batch_size,
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 1000,
            state = Vector{Float32} => (ns,),
        ),
    )

    stop_condition = StopAfterStep(trajectory_num, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook, "BCQD <-> CartPole ($type dataset)")
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_BCQD_CartPole(medium)`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_BCQD_CartPole_medium.png") #hide

# ![](assets/JuliaRL_BCQD_CartPole.png)
