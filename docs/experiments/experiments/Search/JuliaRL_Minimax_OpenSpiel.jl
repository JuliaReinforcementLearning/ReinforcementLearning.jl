function Experiment(::Val{:JuliaRL}, ::Val{:Minimax}, ::Val{:OpenSpiel}, game;)
    env = OpenSpielEnv(string(game))
    agents = MultiAgentManager(
        NamedPolicy(0 => MinimaxPolicy()),
        NamedPolicy(1 => MinimaxPolicy()),
    )
    hooks = MultiAgentHook(0 => TotalRewardPerEpisode(), 1 => TotalRewardPerEpisode())
    description = """
      # Play `$game` in OpenSpiel with Minimax
      """
    Experiment(agents, env, StopAfterEpisode(1), hooks, description)
end
