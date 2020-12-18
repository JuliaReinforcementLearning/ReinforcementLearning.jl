@testset "atari" begin
    atari_env_names = ReinforcementLearningEnvironments.list_atari_rom_names()
    atari_env_names = filter(x -> x ∉ ["pacman", "surround"], atari_env_names)

    @testset "basic tests" begin
        for x in atari_env_names
            env = AtariEnv(x)
            # RLBase.test_interfaces!(env)
            RLBase.test_runnable!(env)
        end
    end

    @testset "seed" begin
        env = AtariEnv(; name = "pong", seed = 456)
        old_states = []
        actions = [rand(action_space(env)) for i in 1:10, j in 1:100]

        for i in 1:10
            for j in 1:100
                env(actions[i, j])
                push!(old_states, copy(state(env)))
            end
            reset!(env)
        end

        env = AtariEnv(; name = "pong", seed = 456)
        new_states = []
        for i in 1:10
            for j in 1:100
                env(actions[i, j])
                push!(new_states, copy(state(env)))
            end
            reset!(env)
        end

        @test old_states == new_states
    end

    @testset "frame_skip" begin
        env = AtariEnv(; name = "pong", frame_skip = 4, seed = 456)
        states = []
        actions = [rand(action_space(env)) for _ in 1:100]

        for i in 1:100
            env(actions[i])
            push!(states, copy(state(env)))
        end

        env = AtariEnv(; name = "pong", frame_skip = 1, seed = 456)
        for i in 1:100
            env(actions[i])
            env(actions[i])
            env(actions[i])
            s1 = copy(state(env))
            env(actions[i])
            s2 = copy(state(env))
            @test states[i] == max.(s1, s2)
        end
    end

    @testset "repeat_action_probability" begin
        env = AtariEnv(; name = "pong", repeat_action_probability = 1.0, seed = 456)
        states = []
        actions = [rand(action_space(env)) for _ in 1:100]
        for i in 1:100
            env(actions[i])
            push!(states, copy(state(env)))
        end

        env = AtariEnv(; name = "pong", repeat_action_probability = 1.0, seed = 456)
        for i in 1:100
            env(actions[1])
            @test states[i] == state(env)
        end
    end

    @testset "max_num_frames_per_episode" begin
        for i in 1:10
            env = AtariEnv(; name = "pong", max_num_frames_per_episode = i, seed = 456)
            for _ in 1:i
                env(1)
            end
            @test true == is_terminated(env)
        end
    end
end
