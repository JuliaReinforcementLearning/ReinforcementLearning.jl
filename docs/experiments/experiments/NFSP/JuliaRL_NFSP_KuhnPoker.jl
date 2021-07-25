# --- 
# title: JuliaRL\_NFSP\_KuhnPoker 
# cover: assets/logo.svg 
# description: NFSP applied to KuhnPokerEnv 
# date: 2021-07-25
# author: "[Peter Chen](https://github.com/peterchen96)" 
# --- 

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses

mutable struct ResultNEpisode <: AbstractHook
    step_counter::Int
    eval_every::Int
    episode::Vector{Int}
    results::Vector{Float64}
end

function (hook::ResultNEpisode)(::PostEpisodeStage, policy, env)
    hook.step_counter += 1
    if hook.step_counter % hook.eval_every == 0
        push!(hook.episode, hook.step_counter)
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

    # Encode the KuhnPokerEnv's states for training.
    env = KuhnPokerEnv()
    wrapped_env = StateTransformedEnv(
        env;
        state_mapping = s -> [findfirst(==(s), state_space(env)) / length(state_space(env))], # for normalization
        state_space_mapping = ss -> [[findfirst(==(s), state_space(env)) / length(state_space(env))] for s in state_space(env)]
        )

    # set parameters for NFSPAgentManager
    nfsp = NFSPAgentManager(wrapped_env;
            η = 0.1,
            _device = Flux.cpu,
            Optimizer = Flux.Descent,
            rng = StableRNG(seed),
            batch_size = 128,
            learn_freq = 128,
            min_buffer_size_to_learn = 1000,
            hidden_layers = (128),
        
            # Reinforcement Learning(RL) agent parameters
            rl_loss_func = mse,
            rl_learning_rate = 0.01,
            replay_buffer_capacity = 200_000,
            ϵ_start = 0.06,
            ϵ_end = 0.001,
            ϵ_decay = 3_000_000,
            discount_factor = 1.0f0,
            update_target_network_freq = 1000,
        
            # Supervisor Learning(SL) agent parameters
            sl_learning_rate = 0.01,
            reservoir_buffer_capacity = 2_000_000,
            )

    stop_condition = StopAfterEpisode(3_000_000, is_show_progress=!haskey(ENV, "CI"))
    hook = ResultNEpisode(0, 10_000, [], [])

    Experiment(nfsp, wrapped_env, stop_condition, hook, "# run NFSP on KuhnPokerEnv")
end

#+ tangle=false
ENV["GKSwstype"]="nul" 
using Plots
ex = E`JuliaRL_NFSP_KuhnPoker`
run(ex)
plot(ex.hook.episode, ex.hook.results, xaxis=:log, xlabel="episode", ylabel="nash_conv")

savefig("assets/JuliaRL_NFSP_KuhnPoker.png")#hide

# ![](assets/JuliaRL_NFSP_KuhnPoker.png)