# ---
# title: JuliaRL\_IDQN\_MPESimple
# cover:
# description: INdependent DQN applied to MPE simple spread
# date: 2023-03-11
# author: "[Panajiotis Keßler](mailto:ppanajiotis.christoforidis@gmail.com)"
# ---

using ReinforcementLearning
using StableRNGs
using Statistics
using Flux
using Flux: glorot_uniform

using Plots
using PyCall
using ArgParse
using Random


include("experiment_hooks.jl")

np = pyimport("numpy")

s = ArgParseSettings()

@add_arg_table s begin
    "episodes"
    help = "Number of epochs"
    arg_type = Int
    default = 10
end
arguments = parse_args(s)


# =============================================================

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:IDQN},
    ::Val{:simpleSpread};
    seed=123,
    n=1,
    γ=0.99f0,
)
    rng = StableRNG(seed)
    env = discrete2standard_discrete(ReinforcementLearning.PettingzooEnv("mpe.simple_spread_v2"; seed=seed))
    ns, na = length(state(env)), length(action_space(env))
    create_policy() = QBasedPolicy(
        learner=DQNLearner(
            approximator=Approximator(
                model=TwinNetwork(
                    Chain(
                        Dense(ns, 128, relu; init=glorot_uniform(rng)),
                        Dense(128, 128, relu; init=glorot_uniform(rng)),
                        Dense(128, na; init=glorot_uniform(rng)),
                    );
                    sync_freq=100
                ),
                optimiser=ADAM(),
            ),
            n=n,
            γ=γ,
            is_enable_double_DQN=true,
            loss_func=Flux.Losses.huber_loss,
            rng=rng,
        ),
        explorer=EpsilonGreedyExplorer(
            kind=:exp,
            ϵ_stable=0.01,
            decay_steps=500,
            rng=rng,
        ),
    )



    policy = MultiAgentManager(
        Dict(
            player => Agent(
                policy = create_policy(),
                trajectory = Trajectory(
                    container=CircularArraySARTTraces(
                        capacity=1000,
                        state=Float32 => (ns,),
                    ),
                    sampler=NStepBatchSampler{SS′ART}(
                        n=n,
                        γ=γ,
                        batch_size=32,
                        rng=rng
                    ),
                    controller=InsertSampleRatioController(
                        threshold=100,
                        n_inserted=-1
                    )
                ),
            ) for player in players(env)
        ),
        current_player(env)
    )
   
   
    stop_condition = StopAfterEpisode(arguments["episodes"], is_show_progress=!haskey(ENV, "CI"))
    
    hook = MeanSTDRewardHook(0, 1, 10, [], Float32[], Float32[])


    Experiment(policy, env, stop_condition, hook)
end


ex = E`JuliaRL_IDQN_simpleSpread`

run(ex)
rewards = ex.hook.rewards
dev = ex.hook.std
plot(ex.hook.rewards, ribbon=ex.hook.std, fillalpha=.45, label="Mean reward", show=true, legend=:outerbottom)
savefig("./IDQN_mpe_simple_spread_$(arguments["episodes"]).png")
