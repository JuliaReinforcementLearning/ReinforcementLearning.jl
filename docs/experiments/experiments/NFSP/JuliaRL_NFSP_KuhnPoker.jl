# --- 
# title: JuliaRL\_NFSP\_KuhnPoker 
# cover: assets/logo.svg 
# description: NFSP applied to KuhnPokerEnv 
# date: 2021-08-07
# author: "[Peter Chen](https://github.com/peterchen96)" 
# --- 

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses

mutable struct KuhnNFSPHook <: AbstractHook
    eval_freq::Int
    episode_counter::Int
    episode::Vector{Int}
    results::Vector{Float64}
end

function (hook::KuhnNFSPHook)(::PostEpisodeStage, policy, env)
    hook.episode_counter += 1
    if hook.episode_counter % hook.eval_freq == 0
        push!(hook.episode, hook.episode_counter)
        push!(hook.results, RLZoo.nash_conv(policy, env))
    end
end

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:NFSP},
    ::Val{:KuhnPoker},
    ::Nothing;
    seed = 123,
)
    rng = StableRNG(seed)
    
    ## Encode the KuhnPokerEnv's states for training.
    env = KuhnPokerEnv()
    wrapped_env = StateTransformedEnv(
        env;
        state_mapping = s -> [findfirst(==(s), state_space(env))],
        state_space_mapping = ss -> [[findfirst(==(s), state_space(env))] for s in state_space(env)]
        )
    player = 1 # or 2
    ns, na = length(state(wrapped_env, player)), length(action_space(wrapped_env, player))

    ## construct rl_agent(use `DQN`) and sl_agent(use `BehaviorCloningPolicy`)
    rl_agent = Agent(
        policy = QBasedPolicy(
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 64, relu; init = glorot_normal(rng)),
                        Dense(64, na; init = glorot_normal(rng))
                    ) |> cpu,
                    optimizer = Descent(0.01),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 64, relu; init = glorot_normal(rng)),
                        Dense(64, na; init = glorot_normal(rng))
                    ) |> cpu,
                ),
                γ = 1.0f0,
                loss_func = huber_loss,
                batch_size = 128,
                update_freq = 128,
                min_replay_history = 1000,
                target_update_freq = 1000,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :linear,
                ϵ_init = 0.06,
                ϵ_stable = 0.001,
                decay_steps = 1_000_000,
                rng = rng,
            ),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 200_000,
            state = Vector{Int} => (ns, ),
        ),
    )

    sl_agent = Agent(
        policy = BehaviorCloningPolicy(;
            approximator = NeuralNetworkApproximator(
                model = Chain(
                        Dense(ns, 64, relu; init = glorot_normal(rng)),
                        Dense(64, na; init = glorot_normal(rng))
                    ) |> cpu,
                optimizer = Descent(0.01),
            ),
            explorer = WeightedSoftmaxExplorer(),
            batch_size = 128,
            min_reservoir_history = 1000,
            rng = rng,
        ),
        trajectory = ReservoirTrajectory(
            2_000_000;# reservoir capacity
            rng = rng,
            :state => Vector{Int},
            :action => Int,
        ),
    )

    ## set parameters and initial NFSPAgentManager
    η = 0.1 # anticipatory parameter
    nfsp = NFSPAgentManager(
        Dict(
            (player, NFSPAgent(
                deepcopy(rl_agent),
                deepcopy(sl_agent),
                η,
                rng,
                128, # update_freq
                0, # initial update_step
                true, # initial NFSPAgent's training mode
            )) for player in players(wrapped_env) if player != chance_player(wrapped_env)
        )
    )

    stop_condition = StopAfterEpisode(1_200_000, is_show_progress=!haskey(ENV, "CI"))
    hook = KuhnNFSPHook(10_000, 0, [], [])

    Experiment(nfsp, wrapped_env, stop_condition, hook, "# run NFSP on KuhnPokerEnv")
end

#+ tangle=false
using Plots
ex = E`JuliaRL_NFSP_KuhnPoker`
run(ex)
plot(ex.hook.episode, ex.hook.results, xaxis=:log, xlabel="episode", ylabel="nash_conv")

savefig("assets/JuliaRL_NFSP_KuhnPoker.png")#hide

# ![](assets/JuliaRL_NFSP_KuhnPoker.png)