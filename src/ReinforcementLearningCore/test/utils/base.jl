@testset "base" begin

    @testset "select_last_dim" begin
        xs = 1:3

        # scalar
        @test select_last_dim(xs, 2) == 2

        # 1d
        @test select_last_dim(xs, [3, 2, 1]) == [3, 2, 1]

        # 2d
        xs = [1 2; 3 4]
        @test select_last_dim(xs, [2, 1]) == [2 1; 4 3]
    end

    @testset "select_last_frame" begin
        xs = 1:3
        @test select_last_frame(xs) == 3

        xs = [1 2; 3 4]
        @test select_last_frame(xs) == [2, 4]
    end

    @testset "consecutive_view" begin
        xs = 1:5

        @test consecutive_view(xs, [2, 3]) == [2, 3]
        @test consecutive_view(xs, [2, 3]; n_stack = 2) == hcat([1, 2], [2, 3])
        @test consecutive_view(xs, [2, 3]; n_horizon = 3) == hcat([2, 3, 4], [3, 4, 5])
        @test consecutive_view(xs, [2, 3]; n_stack = 2, n_horizon = 3) ==
              hcat(
            hcat([1, 2], [2, 3], [3, 4]), # frames at index of 2
            hcat([2, 3], [3, 4], [4, 5]), # frames at index of 3
        ) |> x -> reshape(x, 2, 3, 2)
    end

    @testset "find_all_max" begin
        @test find_all_max([-Inf, -Inf, -Inf]) == (-Inf, [1, 2, 3])
        @test find_all_max([-Inf, -Inf, -Inf], [true, false, true]) == (-Inf, [1, 3])

        @test find_all_max([Inf, Inf, Inf]) == (Inf, [1, 2, 3])
        @test find_all_max([Inf, Inf, Inf], [true, true, false]) == (Inf, [1, 2])

        @test find_all_max([Inf, 0, Inf]) == (Inf, [1, 3])
        @test find_all_max([Inf, 0, Inf], [false, true, false]) == (0, [2])

        @test find_all_max([0, 1, 2, 1, 2, 1, 0]) == (2, [3, 5])
        @test find_all_max([0, 1, 2, 1, 2, 1, 0], Bool[1, 1, 0, 0, 0, 1, 1]) == (1, [2, 6])
    end

    @testset "sum_tree" begin
        t = SumTree(8)

        @test RLCore.capacity(t) == 8

        for i in 1:4
            push!(t, i)
        end

        @test length(t) == 4
        @test size(t) == (4,)

        for i in 5:16
            push!(t, i)
        end

        @test length(t) == 8
        @test size(t) == (8,)
        @test t == 9:16

        t[:] .= 1
        @test t == ones(8)
        @test all([get(t, v)[1] == i for (i, v) in enumerate(0.5:1.0:8)])

        empty!(t)
        @test RLCore.capacity(t) == 8
        @test length(t) == 0
    end

    @testset "flatten_batch" begin
        x = rand(2, 3, 4)
        y = flatten_batch(x)
        @test size(y) == (2, 12)
    end

    @testset "discount_rewards" begin
        reward, γ = [1.0], 0.5
        @test discount_rewards(reward, γ) ≈ reward
        @test discount_rewards(reward, γ; init = 2.0) ≈ [2.0]

        reward, γ = Float64[1, 2, 3], 0.5

        @test discount_rewards(reward, γ) ≈ [2.75, 3.5, 3.0]
        @test discount_rewards(reward, γ; init = 4.0) ≈ [3.25, 4.5, 5.0]

        @test discount_rewards(reward, γ; terminal = [false true false], init = 2.0) ≈
              [2.0, 2.0, 4.0]
        @test discount_rewards(reward, γ; terminal = [true false true], init = 2.0) ≈
              [1.0, 3.5, 3.0]

        # type stable
        reward, γ = [1, 2, 3], 0.5f0
        @test eltype(discount_rewards(reward, γ)) == Float32

        # 2D
        reward, γ = reshape(1:9, 3, 3), 0.5
        init = [-2.0, 0.0, 2.0]
        # for 2d rewards, the keyword argument of `dim` must be either `1` or `2`
        @test_throws MethodError discount_rewards(reward, γ)
        @test discount_rewards(reward, γ; dims = 1) ≈
              [2.75 8.0 13.25; 3.5 8.0 12.5; 3.0 6.0 9.0]
        @test discount_rewards(reward, γ; dims = 2) ≈
              [4.75 7.5 7.0; 6.5 9.0 8.0; 8.25 10.5 9.0]
        @test discount_rewards(reward, γ; init = init, dims = 1) ≈
              [2.5 8.0 13.5; 3.0 8.0 13.0; 2.0 6.0 10.0]
        @test discount_rewards(reward, γ; init = init, dims = 2) ≈
              [4.5 7.0 6.0; 6.5 9.0 8.0; 8.5 11.0 10]

        terminal = [false true false; true false true; false true false]
        @test discount_rewards(reward, γ; dims = 1, terminal = terminal) ≈
              [2.0 4.0 11.0; 2.0 8.0 8.0; 3.0 6.0 9.0]
        @test discount_rewards(reward, γ; dims = 1, terminal = terminal, init = init) ≈
              [2.0 4.0 11.0; 2.0 8.0 8.0; 2.0 6.0 10.0]
        @test discount_rewards(reward, γ; dims = 2, terminal = terminal, init = init) ≈
              [3.0 4.0 6.0; 2.0 9.0 8.0; 6.0 6.0 10.0]
    end

    @testset "discount_rewards_reduced" begin
        reward, γ = [1.0], 0.5
        @test discount_rewards_reduced(reward, γ) ≈ 1.0

        reward, γ = Float64[1, 2, 3], 0.5
        @test discount_rewards_reduced(reward, γ) ≈ 2.75
        @test discount_rewards_reduced(reward, γ; init = 4.0) ≈ 3.25
        @test discount_rewards_reduced(reward, γ; terminal = [false, true, false]) ≈ 2.0
        @test discount_rewards_reduced(
            reward,
            γ;
            terminal = [false, true, false],
            init = 4.0,
        ) ≈ 2.0

        reward, γ = reshape(1:9, 3, 3), 0.5
        init = [-2.0, 0.0, 2.0]
        terminal = [false true false; true false true; false true false]

        # for reward of 2D, `dims` must be provided
        @test_throws Exception discount_rewards_reduced(reward, γ)

        @test discount_rewards_reduced(reward, γ; dims = 1) ≈ [2.75, 8.0, 13.25]
        @test discount_rewards_reduced(reward, γ; dims = 2) ≈ [4.75, 6.5, 8.25]
        @test discount_rewards_reduced(
            reward,
            γ;
            dims = 1,
            terminal = terminal,
            init = init,
        ) ≈ [2.0, 4.0, 11.0]
        @test discount_rewards_reduced(
            reward,
            γ;
            dims = 2,
            terminal = terminal,
            init = init,
        ) ≈ [3.0, 2.0, 6.0]
    end

    @testset "generalized_advantage_estimation" begin
        reward, values, γ, λ = [1.0], [2.0, 3.0], 0.5, 0.3
        @test generalized_advantage_estimation(reward, values, γ, λ) ≈ [0.5]

        reward, values, γ, λ = [1.0, 1.0], [1, 2, 3], 0.5, 0.3
        @test generalized_advantage_estimation(reward, values, γ, λ) ≈ [1.075, 0.5]

        reward, values, γ, λ = Float64[1, 2, 3], [1, 2, 3, 4], 0.5, 0.3
        @test generalized_advantage_estimation(reward, values, γ, λ) ≈ [1.27, 1.8, 2]

        @test generalized_advantage_estimation(
            reward,
            values,
            γ,
            λ;
            terminal = [true false true],
        ) ≈ [0.0, 1.5, 0.0]

        # type stable
        reward, values, γ, λ = [1, 2, 3], [1, 2, 3, 4], 0.5f0, 0.5f0
        @test eltype(generalized_advantage_estimation(reward, values, γ, λ)) == Float32

        # 2D
        reward, values, γ, λ = reshape(1:9, 3, 3), reshape(1:12, 4, 3), 0.5, 0.3
        # for 2d rewards, the keyword argument of `dim` must be either `1` or `2`
        @test_throws MethodError generalized_advantage_estimation(reward, values, γ, λ)
        @test generalized_advantage_estimation(reward, values, γ, λ; dims = 1) ≈
              [1.27 2.4425 3.615; 1.8 2.95 4.1; 2.0 3.0 4.0]

        values = reshape(1:12, 3, 4)
        @test generalized_advantage_estimation(reward, values, γ, λ; dims = 2) ≈
              [2.6375 4.25 5.0; 3.22375 4.825 5.5; 3.81 5.4 6.0]

        reward, values, γ, λ = reshape(1:9, 3, 3), reshape(1:12, 4, 3), 0.5, 0.3
        terminal = [false true false; true false true; false true false]

        @test generalized_advantage_estimation(
            reward,
            values,
            γ,
            λ;
            dims = 1,
            terminal = terminal,
        ) ≈ [1.0 -1.0 2.7; 0.0 2.35 -2.0; 2.0 -1.0 4.0]

        values = reshape(1:12, 3, 4)
        @test generalized_advantage_estimation(reward, values, γ, λ; dims = 2) ≈
              [2.6375 4.25 5.0; 3.22375 4.825 5.5; 3.81 5.4 6.0]
    end

end
