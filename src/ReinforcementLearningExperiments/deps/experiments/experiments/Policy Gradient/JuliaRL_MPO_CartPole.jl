using ReinforcementLearning
using Flux, Random, StableRNGs

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:MPO_continuous},
    ::Val{:CartPole},
    ::Nothing;
    save_dir=nothing,
    seed=123
)
    rng = StableRNG(seed)
    env = ActionTransformedEnv(CartPoleEnv(continuous = true), action_mapping = x->tanh(only(x)))
    seed!(env, seed)
    #continuous with diagonal covariance
    policy = MPOPolicy(
        actor = Approximator(GaussianNetwork(
            Chain(Dense(4, 64, tanh), Dense(64,64,tanh)),
            Dense(64, 1),
            Dense(64, 1)), ADAM(3f-4)),
        qnetwork1 = Approximator(Chain(Dense(5, 64, gelu), Dense(64,64,gelu), Dense(64,1)), ADAM(3f-4)),
        qnetwork2 = Approximator(Chain(Dense(5, 64, gelu), Dense(64,64,gelu), Dense(64,1)), ADAM(3f-4)),
        action_sample_size = 32,
        rng = rng,
        ϵμ = 0.1f0, 
        ϵΣ = 1f-2,
        ϵ = 0.1f0)
    
    agent = Agent(
        policy = policy, 
        trajectory = Trajectory(
            CircularArraySARTTraces(capacity = 1000, state = Float32 => (4,), action = Float32 => (1,)), 
            MetaSampler(
                actor = MultiBatchSampler(BatchSampler{(:state,)}(32), 10),
                critic = MultiBatchSampler(BatchSampler{SS′ART}(32), 1000)
            ),
            InsertSampleRatioController(ratio = 1/1000, threshold = 1000)
        )      
    )

    stop_condition = StopAfterStep(50_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    run(agent, env, stop_condition, hook)
end

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:MPO_discrete},
    ::Val{:CartPole},
    ::Nothing;
    save_dir=nothing,
    seed=123
)
    rng = StableRNG(seed)
    env = ActionTransformedEnv(CartPoleEnv(continuous = false), action_mapping = x -> argmax(x))
    seed!(env, seed)
    #continuous with diagonal covariance
    policy = MPOPolicy(
        actor = Approximator(
            CategoricalNetwork(Chain(Dense(4, 64, tanh), Dense(64,64,tanh), Dense(64,2))),
            ADAM(3f-4)),
        qnetwork1 = Approximator(Chain(Dense(6, 64, gelu), Dense(64,64,gelu), Dense(64,1)), ADAM(3f-4)),
        qnetwork2 = Approximator(Chain(Dense(6, 64, gelu), Dense(64,64,gelu), Dense(64,1)), ADAM(3f-4)),
        action_sample_size = 32,
        rng = rng,
        ϵμ = 1f-1, 
        ϵ = 1f-1)
    
    agent = Agent(
        policy = policy, 
        trajectory = Trajectory(
            CircularArraySARTTraces(capacity = 1000, state = Float32 => (4,), action = Float32 => (2,)), 
            MetaSampler(
                actor = MultiBatchSampler(BatchSampler{(:state,)}(32), 10),
                critic = MultiBatchSampler(BatchSampler{SS′ART}(32), 1000)
            ),
            InsertSampleRatioController(ratio = 1/1000, threshold = 1000)
        )      
    )

    stop_condition = StopAfterStep(50000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    run(agent, env, stop_condition, hook)
end

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:MPO_covariance},
    ::Val{:CartPole},
    ::Nothing;
    save_dir=nothing,
    seed=123
)
    rng = StableRNG(seed)
    env = ActionTransformedEnv(CartPoleEnv(continuous = true), action_mapping = x->tanh(only(x)))
    seed!(env, seed)
    #continuous with diagonal covariance
    policy = MPOPolicy(
        actor = Approximator(CovGaussianNetwork( #using a CovGaussianNetwork makes non sense here because there's one action space dimension. This is only to unit test.
            pre = Chain(Dense(4, 64, tanh), Dense(64,64,tanh)),
            μ = Dense(64, 1),
            Σ = Dense(64, 1)), ADAM(3f-4)),
        qnetwork1 = Approximator(Chain(Dense(5, 64, gelu), Dense(64,64,gelu), Dense(64,1)), ADAM(3f-4)),
        qnetwork2 = Approximator(Chain(Dense(5, 64, gelu), Dense(64,64,gelu), Dense(64,1)), ADAM(3f-4)),
        action_sample_size = 32,
        rng = rng,
        ϵμ = 0.1f0, 
        ϵΣ = 1f-2,
        ϵ = 0.1f0)
    
    agent = Agent(
        policy = policy, 
        trajectory = Trajectory(
            CircularArraySARTTraces(capacity = 1000, state = Float32 => (4,), action = Float32 => (1,)), 
            MetaSampler(
                actor = MultiBatchSampler(BatchSampler{(:state,)}(32), 10),
                critic = MultiBatchSampler(BatchSampler{SS′ART}(32), 1000)
            ),
            InsertSampleRatioController(ratio = 1/1000, threshold = 1000)
        )      
    )

    stop_condition = StopAfterStep(50_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    run(agent, env, stop_condition, hook)
end

using Plots
ex = E`JuliaRL_MPO_discrete_Cartpole`
run(ex)
plot(ex.hook.episodes, ex.hook.mean_rewards, xlabel="episode", ylabel="mean episode reward", title = "Cartpole Discrete Action Space")

savefig("assets/JuliaRL_MPO_discrete_Cartpole.png")

ex = E`JuliaRL_MPO_continuous_Cartpole`
run(ex)
plot(ex.hook.episodes, ex.hook.mean_rewards, xlabel="episode", ylabel="mean episode reward", title = "Cartpole Continuous Action Space")

savefig("assets/JuliaRL_MPO_continuous_Cartpole.png")

ex = E`JuliaRL_MPO_covariance_Cartpole`
run(ex)
plot(ex.hook.episodes, ex.hook.mean_rewards, xlabel="episode", ylabel="mean episode reward", title = "Cartpole Discrete Action Space with MvGaussian")

savefig("assets/JuliaRL_MPO_covariance_Cartpole.png")