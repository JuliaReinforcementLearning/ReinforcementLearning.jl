"""
NFSP agents trained on Kuhn Poker game.
"""
#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses

mutable struct ResultNEpisode <: AbstractHook
    episode::Vector{Int}
    results
end
recorder = ResultNEpisode([], [])

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:NFSP},
    ::Val{:KuhnPoker},
    ::Nothing;
    seed = 123,
)

    # Encode the KuhnPokerEnv's states for training.
    env = KuhnPokerEnv()
    states = [
        (), (:J,), (:Q,), (:K,),
        (:J, :Q), (:J, :K), (:Q, :J), (:Q, :K), (:K, :J), (:K, :Q),
        (:J, :bet), (:J, :pass), (:Q, :bet), (:Q, :pass), (:K, :bet), (:K, :pass),
        (:J, :pass, :bet), (:J, :bet, :bet), (:J, :bet, :pass), (:J, :pass, :pass),
        (:Q, :pass, :bet), (:Q, :bet, :bet), (:Q, :bet, :pass), (:Q, :pass, :pass),
        (:K, :pass, :bet), (:K, :bet, :bet), (:K, :bet, :pass), (:K, :pass, :pass),
        (:J, :pass, :bet, :pass), (:J, :pass, :bet, :bet), (:Q, :pass, :bet, :pass),
        (:Q, :pass, :bet, :bet), (:K, :pass, :bet, :pass), (:K, :pass, :bet, :bet),
    ] # collect all states
    states_indexes_Dict = Dict((i, j) for (j, i) in enumerate(states))
    wrapped_env = StateTransformedEnv(
            env;
            state_mapping = s -> [states_indexes_Dict[s]],
            state_space_mapping = ss -> [[i] for i in 1:length(states)]
        )

    # set parameters and initial NFSPAgents
    nfsp = NFSPAgents(wrapped_env;
            η = 0.1,
            _device = Flux.cpu,
            Optimizer = Flux.Descent,
            rng = StableRNG(seed),
            batch_size = 128,
            learn_freq = 128,
            min_buffer_size_to_learn = 1000,
            hidden_layers = (128, 128),
        
            # Reinforcement Learning(RL) agent parameters
            rl_loss_func = mse,
            rl_learning_rate = 0.01,
            replay_buffer_capacity = 200_000,
            ϵ_start = 0.06,
            ϵ_end = 0.001,
            ϵ_decay = 20_000_000,
            discount_factor = 1.0f0,
            update_target_network_freq = 19200,
        
            # Supervisor Learning(SL) agent parameters
            sl_learning_rate = 0.01,
            reservoir_buffer_capacity = 2_000_000,
            )

    stop_condition = StopAfterEpisode(10_000_000, is_show_progress=!haskey(ENV, "CI"))
    hook = DoEveryNEpisode(; n = 10_000) do t, nfsp, wrapped_env
            push!(recorder.episode, t)
            push!(recorder.results, RLZoo.nash_conv(nfsp, wrapped_env))
        end
    Experiment(nfsp, wrapped_env, stop_condition, hook, "")
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_NFSP_KuhnPoker`
run(ex)
plot(recorder.episode, recorder.results, xaxis=:log, yaxis=:log)
xlabel!("episode")
ylabel!("nash_conv")

savefig("assets/JuliaRL_NFSP_KuhnPoker.png")#hide

# ![](assets/JuliaRL_NFSP_KuhnPoker.png)