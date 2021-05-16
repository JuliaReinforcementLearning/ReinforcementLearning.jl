function Experiment(
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

    description = """
        # Play `$game` in OpenSpiel with TabularCFRPolicy
        """
    Experiment(π, env, StopAfterStep(300), EmptyHook(), description)
end