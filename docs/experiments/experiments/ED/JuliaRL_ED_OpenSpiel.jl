# --- 
# title: JuliaRL\_ED\_OpenSpiel(kuhn_poker) 
# cover: assets/logo.svg 
# description: play "kuhn_poker" in OpenSpiel with Exploitability Descent(ED) algorithm.
# date: 2021-09-20
# author: "[Peter Chen](https://github.com/peterchen96)" 
# --- 

#+ tangle=false
using ReinforcementLearning
using StableRNGs
using OpenSpiel
using Flux

mutable struct KuhnOpenEDHook <: AbstractHook
    results::Vector{Float64}
end

function (hook::KuhnOpenEDHook)(::PreEpisodeStage, policy, env)
    ## get nash_conv of the current policy.
    push!(hook.results, RLZoo.nash_conv(policy, env))
    
    ## update agents' learning rate.
    for (_, agent) in policy.agents
        agent.learner.optimizer[2].eta = 1.0 / sqrt(length(hook.results))
    end
end

function (hook::KuhnOpenEDHook)(::PostExperimentStage, policy, env)
    reset!(env)

    ## get nash_conv of the latest model.
    push!(hook.results, RLZoo.nash_conv(policy, env))
end

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:ED},
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

    create_network() = Chain(
        Dense(ns, 64, relu;init = glorot_uniform(rng)),
        Dense(64, na;init = glorot_uniform(rng))
    )

    create_learner() = NeuralNetworkApproximator(
        model = create_network(),
        optimizer = Flux.Optimise.Optimiser(WeightDecay(0.001), Descent())
    )

    EDmanager = EDManager(
        Dict(
            player => EDPolicy(
                1 - player, # opponent
                create_learner(), # neural network learner
                WeightedSoftmaxExplorer(), # explorer
            ) for player in players(env) if player != chance_player(env)
        )
    )

    stop_condition = StopAfterEpisode(500, is_show_progress=!haskey(ENV, "CI"))
    hook = KuhnOpenEDHook([])

    Experiment(EDmanager, wrapped_env, stop_condition, hook, "# play OpenSpiel $game with ED algorithm")
end

using Plots
ex = E`JuliaRL_ED_OpenSpiel(kuhn_poker)`
run(ex)
plot(ex.hook.results, xlabel="episode", ylabel="nash_conv")

savefig("assets/JuliaRL_ED_OpenSpiel(kuhn_poker).png")#hide

# ![](assets/JuliaRL_ED_OpenSpiel(kuhn_poker).png)