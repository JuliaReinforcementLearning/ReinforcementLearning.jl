using ReinforcementLearning
using Pkg; Pkg.activate("./src/ReinforcementLearningExperiments")
using Flux, Random, StableRNGs

seed = 123
#=function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:MPO},
    ::Val{:CartPole},
    ::Nothing;
    save_dir=nothing,
    seed=123
)=#
begin
    rng = StableRNG(seed)
    env = ActionTransformedEnv(CartPoleEnv(continuous = true), action_mapping = x->only(x))
    seed!(env, seed)
    #continuous with diagonal covariance
    policy = MPOPolicy(
        policy = Approximator(GaussianNetwork(
            pre = Chain(Dense(4, 64, gelu)),
            μ = Dense(64, 1),
            logσ = Dense(64, 1)), ADAM(3f-4)),
        qnetwork1 = Approximator(Chain(Dense(5, 64, gelu), Dense(64,1)), ADAM(3f-4)),
        qnetwork2 = Approximator(Chain(Dense(5, 64, gelu), Dense(64,1)), ADAM(3f-4)),
        action_sample_size = 32,
        rng = rng,
        ϵμ = 0.01f0, 
        ϵΣ = 1f-2,
        ϵ = 0.1f0)
    
    agent = Agent(
        policy = policy, 
        trajectory = Trajectory(
            CircularArraySARTTraces(capacity = 10000, state = Float32 => (4,), action = Float32 => (1,)), 
            MetaSampler(
                policy = MultiBatchSampler(BatchSampler{(:state,)}(32), 1000),
                critic = MultiBatchSampler(BatchSampler{SS′ART}(32), 1000)
            ),
            InsertSampleRatioController(ratio = 1/10000, threshold = 10000)
        )      
    )

    stop_condition = StopAfterStep(1000_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    run(agent, env, stop_condition, hook)
end