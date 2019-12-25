@testset "utils" begin

@testset "select_last_dim" begin
    @testset "1D" begin
        xs = [1, 2, 3, 4]

        @test select_last_frame(xs) == 4

        @test select_last_dim(xs, 2) == 2
        @test select_last_dim(xs, [2, 1]) == [2, 1]
        @test select_last_dim(xs, [2 1; 4 3]) == [2 1; 4 3]
    end

    @testset "3D" begin
        xs = rand(2, 3, 4)

        @test select_last_frame(xs) == xs[:, :, 4]

        @test select_last_dim(xs, 1) == xs[:, :, 1]
        @test select_last_dim(xs, [2, 3, 1]) == xs[:, :, [2, 3, 1]]
        @test select_last_dim(xs, [1 2; 3 4]) == xs[:, :, [1 2; 3 4]]
    end
end

@testset "consecutive_view" begin
    @testset "1D" begin
        xs = [1, 2, 3, 4]

        @test consecutive_view(xs, [3, 2]) == [3, 2]
        @test consecutive_view(xs, [3, 2]; n_stack=2) == [2 1; 3 2]
        @test consecutive_view(xs, [3, 2]; n_horizon=2) == [3 2; 4 3]
        @test consecutive_view(xs, [3, 2]; n_stack=2, n_horizon=2) == reshape(
            [
                2, 3, 3, 4,
                1, 2, 2, 3
            ], 2, 2, 2)
    end

    @testset "3D" begin
        xs = reshape(repeat(1:4, inner=(4,)), 2, 2, 4)
        #=
        2×2×4 Array{Int64,3}:
        [:, :, 1] =
        1  1
        1  1

        [:, :, 2] =
        2  2
        2  2

        [:, :, 3] =
        3  3
        3  3

        [:, :, 4] =
        4  4
        4  4
        =#

        @test consecutive_view(xs, [3, 2]) == reshape([3, 3, 3, 3, 2, 2, 2, 2], 2, 2, 2)
        @test consecutive_view(xs, [3, 2]; n_stack=2) == reshape(repeat([2,3,1,2]; inner=4), 2, 2, 2, 2)
        @test consecutive_view(xs, [3, 2]; n_horizon=2) == reshape(repeat([3,4,2,3], inner=4), 2, 2, 2, 2)
        @test consecutive_view(xs, [3, 2]; n_stack=2, n_horizon=2) == reshape(repeat([2,3,3,4,1,2,2,3], inner=4), 2, 2, 2, 2, 2)
    end
end

@testset "find_all_max" begin
    @test find_all_max([-Inf, -Inf, -Inf]) == (-Inf, [1, 2, 3])
    @test find_all_max([Inf, Inf, Inf]) == (Inf, [1, 2, 3])
    @test find_all_max([0, 1, 2, 1, 2, 1, 0]) == (2, [3, 5])

    max_val, inds = find_all_max([NaN, NaN, NaN])
    @test isnan(max_val) && inds == [1, 2, 3]

    max_val, inds = find_all_max([0, 1, 2, 1, 2, 1, 0], [true, true, false, false, false, true, true])
    @test max_val == 1
    @test inds == [2, 6]
end

end