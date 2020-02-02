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

end
