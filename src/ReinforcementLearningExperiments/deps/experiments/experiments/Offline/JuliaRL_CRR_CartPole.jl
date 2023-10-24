# ---
# title: JuliaRL\_CRR\_CartPole
# cover: assets/JuliaRL_CRR_CartPole_medium.png
# description: CRR applied to CartPole
# date: 2021-09-17
# author: "[Guoyu Yang](https://github.com/pilgrimygy)"
# ---

#+ tangle=true
using ReinforcementLearningCore, ReinforcementLearningBase, ReinforcementLearningZoo
using ReinforcementLearningEnvironments
using StableRNGs
using Flux
using Flux.Losses

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:CRR},
    ::Val{:CartPole},
    type::AbstractString;
    save_dir = nothing,
    seed = 123,
)
    rng = StableRNG(seed)
    UPDATE_FREQ = 10
    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))
    
    trajectory_num = 10000
    dataset_size = 10000
    batchsize = 64

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
        optimizer = Adam(1e-3),
    )

    agent = Agent(
        policy = OfflinePolicy(
            learner = CRRLearner(
                approximator = create_actor_critic() |> cpu,
                target_approximator = create_actor_critic() |> cpu,
                γ = 0.99f0,
                batchsize = batchsize,
                continuous = false,
                policy_improvement_mode = :exp,
                ratio_upper_bound = 20.0f0,
                β = 1.0f0,
                advantage_estimator = :mean,
                m = 4,
                update_freq = UPDATE_FREQ,
            ),
            dataset = gen_JuliaRL_dataset(:BasicDQN, :CartPole, type; dataset_size = dataset_size),
            continuous = false,
            batchsize = batchsize,
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 1000,
            state = Vector{Float32} => (ns,),
        ),
    )

    stop_condition = StopAfterStep(trajectory_num, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook, "CRR <-> CartPole ($type dataset)")
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_CRR_CartPole(medium)`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_CRR_CartPole_medium.png") #hide

# ![](assets/JuliaRL_CRR_CartPole.png)
