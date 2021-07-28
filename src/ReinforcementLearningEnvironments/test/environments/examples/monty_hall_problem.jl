@testset "MontyHallEnv" begin

    rng = StableRNG(123)
    env = MontyHallEnv(; rng = rng)

    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

    n_win_car = 0
    N = 50_000

    for _ in 1:N
        a = rand(rng, action_space(env))
        env(a)
        env(a)
        if reward(env) == RLEnvs.REWARD_OF_CAR
            n_win_car += 1
        end
        reset!(env)
    end

    @test isapprox(n_win_car / N, 1 / 3; atol = 0.01)

end
