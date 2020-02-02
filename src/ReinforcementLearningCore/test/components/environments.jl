@testset "environments" begin

    @testset "test API" begin

        for env in [CartPoleEnv(), WrappedEnv(CloneStatePreprocessor(), CartPoleEnv())]
            reset!(env)
            action_space = get_action_space(env)
            observation_space = get_observation_space(env)

            obs = observe(env)

            for _ in 1:1000
                if get_terminal(obs)
                    reset!(env)
                    obs = observe(env)
                end
                @test get_state(obs) ∈ observation_space
                action = rand(action_space)
                env(action)
                obs = observe(env)
            end
        end

    end

end
