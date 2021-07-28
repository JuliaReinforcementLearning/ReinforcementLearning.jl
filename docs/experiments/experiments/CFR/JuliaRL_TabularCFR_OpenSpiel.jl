# ---
# title: JuliaRL\_TabularCFR\_OpenSpiel(kuhn_poker)
# cover: assets/logo.svg
# description: TabularCFR applied to OpenSpiel(kuhn_poker)
# date: 2021-05-22
# author: "[Jun Tian](https://github.com/findmyway)"
# ---

#+ tangle=false
using ReinforcementLearning
using OpenSpiel

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:TabularCFR},
    ::Val{:OpenSpiel},
    game;
    n_iter = 300,
    seed = 123,
)
    env = OpenSpielEnv(game)
    rng = StableRNG(seed)
    π = TabularCFRPolicy(; rng = rng)

    description = "# Play `$game` in OpenSpiel with TabularCFRPolicy"
    Experiment(π, env, StopAfterStep(300, is_show_progress=!haskey(ENV, "CI")), EmptyHook(), description)
end

ex = E`JuliaRL_TabularCFR_OpenSpiel(kuhn_poker)`
run(ex)