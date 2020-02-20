@testset "OpenSpielEnv" begin

for name in ["tic_tac_toe", "kuhn_poker", "goofspiel(imp_info=True,num_cards=4,points_order=descending)"]
    env = OpenSpielEnv(name, seed=123)
    get_current_player(env)
    get_num_players(env)
    get_observation_space(env)
    get_action_space(env)
    DynamicStyle(env)

    obs = observe(env)
    obs_0 = observe(env, 0)
    obs_1 = observe(env, 1)
    ActionStyle(obs_0)
    get_legal_actions_mask(obs_0)
    get_legal_actions_mask(obs_1)
    get_legal_actions(obs_0)
    get_legal_actions(obs_1)
    get_terminal(obs_0)
    get_terminal(obs_1)
    get_reward(obs_0)
    get_reward(obs_1)
    get_state(obs_0)
    get_state(obs_1)
    get_invalid_action(obs_0)

    Random.seed!(env, 456)
    reset!(env)

    while true
        obs = observe(env)
        get_terminal(obs) && break
        action = rand(get_legal_actions(obs))
        env(action)
    end
    @test true
end
end