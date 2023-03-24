using ReinforcementLearning
using StableRNGs
using Statistics
using Flux
using Flux: glorot_uniform

using Plots
using PyCall
using ArgParse
using Distributions

using Random

include("experiment_hooks.jl")

np = pyimport("numpy")

s = ArgParseSettings()

@add_arg_table s begin
    "episodes"
    help = "Number of epochs"
    arg_type = Int
    default = 5_000
end
arguments = parse_args(s)


# =========================== def exp. ===========================

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:MADDPG},
    ::Val{:SimpleSpread};
    seed=123,
    n=25,
    γ=0.99f0
)
    rng = StableRNG(seed)
    env = ReinforcementLearning.PettingZooEnv("mpe.simple_spread_v2"; max_cycles=n, seed=seed, continuous_actions=true)
    na = (player) -> length(action_space(env, player).domains)
    l_border = [i.left for i in action_space(env).domains]
    r_border = [i.right for i in action_space(env).domains]

    init = glorot_uniform(rng)
    critic_dim = sum(length(state(env, p)) + na(p) for p in players(env))
    
    
    create_actor(player) = Chain(
        Dense(length(state(env, player)), 64, relu; init = init),
        Dense(64, 64, relu; init = init),
        Dense(64, na(player); init = init),
        x -> max.(min.(x, r_border), l_border)
    )

    create_critic(critic_dim) = Chain(
        Dense(critic_dim, 64, relu; init = init),
        Dense(64, 64, relu; init = init),
        Dense(64, 1; init = init),
    ) 
    create_policy(player) = DDPGPolicy(
            actor = Approximator(
                model = TwinNetwork(
                    create_actor(player),
                    ρ = 0.995f0
                ),
                optimiser = Flux.Optimise.Optimiser(ClipNorm(0.5), ADAM(1e-2)),
              ),
            critic = Approximator(
                model = TwinNetwork(
                    create_critic(critic_dim),
                    ρ = 0.995f0
                ),
                optimiser = Flux.Optimise.Optimiser(ClipNorm(0.5), ADAM(1e-2)),
            ),
            γ = 0.95f0,
            na = na(player),
            start_steps = 0,
            start_policy = env -> rand(Distributions.Uniform(0, 1), na(player)),
            update_after = 512 * 25, # batch_size * env.max_steps
            act_upper_limit = 1.0,
            act_lower_limit = 0.0,
            act_noise = 9e-2,
        )
    create_trajectory(player) = Trajectory(
            container=CircularArraySARTTraces(
                capacity=2,
                state=Float32 => (length(state(env, player)),),
                action=Float64 => (length(action_space(env, player).domains),)
            ),
            sampler=NStepBatchSampler{SS′ART}(
                n=1,
                γ=γ,
                batch_size=n,
                rng=rng
            ),
            controller=InsertSampleRatioController(
                threshold=128
            )
        )

    agents = MADDPGManager(
        Dict(
            player => Agent(
                create_policy(player),
                create_trajectory(player),
            ) for player in players(env)
        ),
        1, # update_freq
        0 # initial update_step
    )

    stop_condition = StopAfterEpisode(arguments["episodes"], is_show_progress=!haskey(ENV, "CI"))
    
    hook = MeanSTDRewardHook(0, 1, 10, [], Float32[], Float32[])
    Experiment(agents, env, stop_condition, hook)
end


ex = E`JuliaRL_MADDPG_SimpleSpread`
run(ex)


rewardHook = ex.hook
#plot(ex.hook.rewards)
rewards = rewardHook.rewards
dev = rewardHook.std
plot(rewards, ribbon=std, fillalpha=.45, label="Mean reward /w sd", show=true, legend=:outerbottom)
savefig("./MADDPG_mpe_simple_spread_$(arguments["episodes"]).png")
