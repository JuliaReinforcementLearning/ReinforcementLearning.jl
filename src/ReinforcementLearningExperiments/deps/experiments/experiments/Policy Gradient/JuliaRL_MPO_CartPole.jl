using ReinforcementLearning
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
    rng = StableRNG(seed)
    env = ActionTransformedEnv(CartPoleEnv(continuous = true), action_mapping = x->only(x))
    seed!(env, seed)
    #continuous with diagonal covariance
    policy = MPOPolicy(
        policy = Approximator(GaussianNetwork(
            pre = Chain(Dense(4, 64, gelu), Dense(64, 64, gelu)),
            μ = Dense(64, 1, tanh),
            logσ = Dense(64, 1)), ADAM()),
        qnetwork1 = Approximator(Chain(Dense(5, 64, gelu), Dense(64,64,gelu), Dense(64,1)), ADAM()),
        qnetwork2 = Approximator(Chain(Dense(5, 64, gelu), Dense(64,64,gelu), Dense(64,1)), ADAM()),
        action_sample_size = 32,
        rng = rng)
    
    agent = Agent(
        policy = policy, 
        trajectory = Trajectory(
            CircularArraySARTTraces(capacity = 1000, state = Float32 => (4,), action = Float32 => (1,)), 
            MetaSampler(
                policy = MultiBatchSampler(BatchSampler{(:state,)}(64), 3),
                critic = MultiBatchSampler(BatchSampler{SS′ART}(64), 5)
            ),
            InsertSampleRatioController(ratio = 5, threshold = 1000)
        )          
    )

    stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    run(agent, env, stop_condition, hook)
