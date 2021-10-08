# ---
# title: JuliaRL\_VMPO\_CartPole
# cover: assets/JuliaRL_VMPO_CartPole.png
# description: VMPO applied to CartPole
# date: 2021-08-25
# author: "[Bo Lu](https://github.com/burmecia)"
# ---

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:VMPO},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    HIDDEN_LAYER = 64
    UPDATE_FREQ = 32
    rng = StableRNG(seed)

    env = CartPoleEnv(; T = Float32, rng = StableRNG(seed + 1))
    ns, na = length(state(env)), length(action_space(env))

    agent = Agent(
        policy = VMPOPolicy(
            approximator = ActorCritic(
                actor = Chain(
                    Dense(ns, HIDDEN_LAYER, relu; init = glorot_uniform(rng)),
                    Dense(HIDDEN_LAYER, na; init = glorot_uniform(rng)),
                ),
                critic = Chain(
                    Dense(ns, HIDDEN_LAYER, relu; init = glorot_uniform(rng)),
                    Dense(HIDDEN_LAYER, 1; init = glorot_uniform(rng)),
                ),
                optimizer = ADAM(3e-4),
            ) |> gpu,
            γ = 0.99f0,
            ϵ_η = 0.02f0,
            ϵ_α = 0.1f0,
            n_epochs = 8,
            update_freq = UPDATE_FREQ,
            rng = rng,
        ),
        trajectory = VMPOTrajectory(
            capacity = UPDATE_FREQ,
            state = Float32 => (ns,),
            action = Int => (),
            reward = Float32 => (),
            terminal = Bool => (),
        ),
    )

    stop_condition = StopAfterStep(50_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()

    Experiment(agent, env, stop_condition, hook, "# VMPO with CartPole")
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_VMPO_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_VMPO_CartPole.png") #hide

# ![](assets/JuliaRL_VMPO_CartPole.png)
