@testset "OpenSpielEnv" begin

    for name in [
        "tic_tac_toe",
        "kuhn_poker",
        "goofspiel(imp_info=True,num_cards=4,points_order=descending)",
    ]
        @info "testing OpenSpiel: $name"
        env = OpenSpielEnv(name)
        get_current_player(env)
        get_actions(env)
        DynamicStyle(env)

        env_0 = SubjectiveEnv(env, 0)
        env_1 = SubjectiveEnv(env, 1)
        ActionStyle(env_0)
        get_legal_actions_mask(env_0)
        get_legal_actions_mask(env_1)
        get_legal_actions(env_0)
        get_legal_actions(env_1)
        get_terminal(env_0)
        get_terminal(env_1)
        get_reward(env_0)
        get_reward(env_1)
        get_state(env_0)
        get_state(env_1)

        reset!(env)

        while true
            get_terminal(env) && break
            action = rand(get_legal_actions(env))
            env(action)
        end
        @test true
    end
end
