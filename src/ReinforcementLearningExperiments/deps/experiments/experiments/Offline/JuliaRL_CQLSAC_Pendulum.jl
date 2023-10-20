# ---
# title: JuliaRL\_SAC\_Pendulum
# cover: assets/JuliaRL_SAC_Pendulum.png
# description: SAC applied to Pendulum
# date: 2021-05-22
# author: "[Roman Bange](https://github.com/rbange)"
# ---

#+ tangle=true
using ReinforcementLearningCore, ReinforcementLearningBase, ReinforcementLearningZoo, ReinforcementLearningEnvironments
using StableRNGs
using Flux
using Flux.Losses
using IntervalSets

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:CQLSAC},
    ::Val{:Pendulum},
    dummy = nothing;
    save_dir=nothing,
    seed=123
)
    rng = StableRNG(seed)
    inner_env = PendulumEnv(T=Float32, rng=rng)
    action_dims = inner_env.n_actions
    A = action_space(inner_env)
    low = A.left
    high = A.right
    ns = length(state(inner_env))
    na = 1

    env = ActionTransformedEnv(
        inner_env;
        action_mapping=x -> low + (x[1] + 1) * 0.5 * (high - low)
    )
    init = Flux.glorot_uniform(rng)

    create_policy_net() = Approximator(
        SoftGaussianNetwork(
            Chain(
                Dense(ns, 30, relu, init=init),
                Dense(30, 30, relu, init=init),
            ),
            Chain(Dense(30, na, init=init)),
            Chain(Dense(30, na, softplus, init=init)),
        ),
        Adam(0.003),
    )

    create_q_net() = TargetNetwork(
        Approximator(
            Chain(
                Dense(ns + na, 30, relu; init=init),
                Dense(30, 30, relu; init=init),
                Dense(30, 1; init=init),
            ),
            Adam(0.003),),
        ρ = 0.99f0
    )
    trajectory= Trajectory(
            CircularArraySARTSTraces(capacity = 10000, state = Float32 => (ns,), action = Float32 => (na,)),
            BatchSampler{SS′ART}(64),
            InsertSampleRatioController(ratio = Inf, threshold = 0)) # There are no insertions in Offline RL, the controller is not used.

    agent = OfflineAgent(
        policy = CQLSACPolicy(
            sac = SACPolicy(
                policy=create_policy_net(),
                qnetwork1=create_q_net(),
                qnetwork2=create_q_net(),
                γ=0.99f0,
                α=0.2f0,
                start_steps=1000,
                start_policy=RandomPolicy(-1.0 .. 1.0; rng=rng),
                automatic_entropy_tuning=true,
                lr_alpha=0.003f0,
                action_dims=action_dims,
                rng=rng,
                device_rng= rng
            )            
        ),
        trajectory = trajectory,
        behavior_agent = Agent(RandomPolicy(-1.0 .. 1.0; rng=rng), trajectory)
    )

    stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode() 
    Experiment(agent, env, stop_condition, hook) |> run
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_SAC_Pendulum`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_SAC_Pendulum.png") #hide

# ![](assets/JuliaRL_SAC_Pendulum.png)
