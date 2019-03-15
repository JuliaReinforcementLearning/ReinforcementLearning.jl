function basic_env_test(env, n=100)
    os = observation_space(env)
    as = action_space(env)
    @test reset!(env) == nothing
    for _ in 1:n
        a = rand(as)
        @test a in as
        obs, reward, isdone = interact!(env, a)
        @test obs in os
        if isdone
            reset!(env)
        end
    end
end
