# ---
# title: JuliaRL\_MPO\_Cartpole
# cover:
# description: Solving Cartpole with MPO with a Discrete or a Continuous action space.
# date: 2022-12-20
# author: "[Henri Dehaybe](https://github.com/HenriDeh)"
# ---

using ReinforcementLearningCore
using Flux, Random, StableRNGs

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:MPOContinuous},
    ::Val{:CartPole},
    dummy = nothing;
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
        ϵμ = 0.01f0, 
        ϵΣ = 1f-4,
        ϵ = 0.1f0)
    
    agent = Agent(
        policy = policy, 
        trajectory = Trajectory(
            CircularArraySARTTraces(capacity = 1000, state = Float32 => (4,), action = Float32 => (1,)), 
            MetaSampler(
                actor = MultiBatchSampler(BatchSampler{(:state,)}(32), 10),
                critic = MultiBatchSampler(BatchSampler{SS′ART}(32), 2000)
            ),
            InsertSampleRatioController(ratio = 1/1000, threshold = 1000)
        )      
    )

    stop_condition = StopAfterStep(50_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook)
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:MPODiscrete},
    ::Val{:CartPole},
    dummy = nothing;
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
        ϵμ = 1f-2, 
        ϵ = 1f-2)
    
    agent = Agent(
        policy = policy, 
        trajectory = Trajectory(
            CircularArraySARTTraces(capacity = 1000, state = Float32 => (4,), action = Float32 => (2,)), 
            MetaSampler(
                actor = MultiBatchSampler(BatchSampler{(:state,)}(32), 10),
                critic = MultiBatchSampler(BatchSampler{SS′ART}(32), 2000)
            ),
            InsertSampleRatioController(ratio = 1/1000, threshold = 1000)
        )      
    )

    stop_condition = StopAfterStep(50000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook)
end



function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:MPOCovariance},
    ::Val{:CartPole},
    dummy = nothing;
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
        ϵμ = 0.01f0, 
        ϵΣ = 1f-4,
        ϵ = 0.01f0)
    
    agent = Agent(
        policy = policy, 
        trajectory = Trajectory(
            CircularArraySARTTraces(capacity = 1000, state = Float32 => (4,), action = Float32 => (1,)), 
            MetaSampler(
                actor = MultiBatchSampler(BatchSampler{(:state,)}(32), 10),
                critic = MultiBatchSampler(BatchSampler{SS′ART}(32), 2000)
            ),
            InsertSampleRatioController(ratio = 1/1000, threshold = 1000)
        )      
    )

    stop_condition = StopAfterStep(50_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook)
end
#+ tangle=false

ex = E`JuliaRL_MPOContinuous_CartPole`
run(ex)

ex = E`JuliaRL_MPODiscrete_CartPole`
run(ex)

ex = E`JuliaRL_MPOCovariance_CartPole`
run(ex)
