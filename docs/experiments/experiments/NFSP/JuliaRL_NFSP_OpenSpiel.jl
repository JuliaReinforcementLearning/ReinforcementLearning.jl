# --- 
# title: JuliaRL\_NFSP\_OpenSpiel(kuhn_poker) 
# cover: assets/logo.svg 
# description: play "kuhn_poker" in OpenSpiel with NFSP
# date: 2021-09-06
# author: "[Peter Chen](https://github.com/peterchen96)" 
# --- 

#+ tangle=false
using ReinforcementLearning
using StableRNGs
using OpenSpiel
using Flux
using Flux.Losses

mutable struct KuhnOpenNFSPHook <: AbstractHook
    eval_freq::Int
    episode_counter::Int
    episode::Vector{Int}
    results::Vector{Float64}
end

function (hook::KuhnOpenNFSPHook)(::PostEpisodeStage, policy, env)
    RLBase.reset!(env)
    hook.episode_counter += 1
    if hook.episode_counter % hook.eval_freq == 0
        push!(hook.episode, hook.episode_counter)
        push!(hook.results, RLZoo.nash_conv(policy, env))
    end
end

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:NFSP},
    ::Val{:OpenSpiel},
    game;
    seed = 123,
)
    rng = StableRNG(seed)
    
    env = OpenSpielEnv(game)
    wrapped_env = ActionTransformedEnv(
        env,
        action_mapping = a -> RLBase.current_player(env) == chance_player(env) ? a : Int(a - 1),
        action_space_mapping = as -> RLBase.current_player(env) == chance_player(env) ? 
            as : Base.OneTo(num_distinct_actions(env.game)),
    )
    wrapped_env = DefaultStateStyleEnv{InformationSet{Array}()}(wrapped_env)
    player = 0 # or 1
    ns, na = length(state(wrapped_env, player)), length(action_space(wrapped_env, player))

    ## construct rl_agent(use `DQN`) and sl_agent(use `BehaviorCloningPolicy`)
    rl_agent = Agent(
        policy = QBasedPolicy(
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; init = glorot_normal(rng)),
                        Dense(128, na; init = glorot_normal(rng))
                    ) |> cpu,
                    optimizer = Descent(0.01),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; init = glorot_normal(rng)),
                        Dense(128, na; init = glorot_normal(rng))
                    ) |> cpu,
                ),
                γ = 1.0f0,
                loss_func = mse,
                batch_size = 128,
                update_freq = 128,
                min_replay_history = 1000,
                target_update_freq = 19500,
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
            state = Vector{Float64} => (ns, ),
        ),
    )

    sl_agent = Agent(
        policy = BehaviorCloningPolicy(;
            approximator = NeuralNetworkApproximator(
                model = Chain(
                        Dense(ns, 128, relu; init = glorot_normal(rng)),
                        Dense(128, na; init = glorot_normal(rng))
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
            :state => Vector{Float64},
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
    hook = KuhnOpenNFSPHook(10_000, 0, [], [])

    Experiment(nfsp, wrapped_env, stop_condition, hook, "# Play kuhn_poker in OpenSpiel with NFSP")
end

using Plots
ex = E`JuliaRL_NFSP_OpenSpiel(kuhn_poker)`
run(ex)
plot(ex.hook.episode, ex.hook.results, xaxis=:log, xlabel="episode", ylabel="nash_conv")

savefig("assets/JuliaRL_NFSP_OpenSpiel(kuhn_poker).png")#hide

# ![](assets/JuliaRL_NFSP_OpenSpiel(kuhn_poker).png)