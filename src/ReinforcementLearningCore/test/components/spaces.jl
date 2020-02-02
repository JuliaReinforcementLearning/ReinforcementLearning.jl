@testset "Spaces" begin

    function test_samples(s, n = 100)
        for _ in 1:n
            @test rand(s) in s
        end
    end

    @testset "DiscreteSpace" begin
        @test_throws ArgumentError DiscreteSpace(0)
        @test_throws MethodError DiscreteSpace(2.2)

        s = DiscreteSpace(3)
        @test length(s) == 3
        @test eltype(s) == Int
        @test element_length(s) == 0
        @test element_size(s) == ()

        @test !(0 in s)
        @test 1 in s
        @test 3 in s
        @test !(4 in s)

        test_samples(s)
    end

    @testset "MultiDiscreteSpace" begin
        @test_throws ArgumentError MultiDiscreteSpace([2, -1, 3])
        @test_throws ArgumentError MultiDiscreteSpace([2, -1, 3])
        @test_throws InexactError MultiDiscreteSpace([2.1, 1.3, 3.0])
        @test_nowarn MultiDiscreteSpace([2, 2, 3])

        s = MultiDiscreteSpace([3, 5, 4])
        @test length(s) == 60
        @test !([0, 0, 1] in s)
        @test [1, 1, 1] in s
        @test !([4, 4, 4] in s)
        @test [3, 3, 3] in s
        @test !([5, 5, 5] in s)

        @test element_length(s) == 3
        @test element_size(s) == (3,)

        test_samples(s)
    end

    @testset "ContinuousSpace" begin
        @test_throws ArgumentError ContinuousSpace(2, 1)

        s = ContinuousSpace(-1.0, 1)
        @test eltype(s) == Float64
        @test !(-2 in s)
        @test -1 in s
        @test 0 in s
        @test 1 in s
        @test !(2 in s)

        test_samples(s)
    end

    @testset "MultiContinuousSpace" begin
        @test_throws ArgumentError MultiContinuousSpace([1, 2], [0, 0])
        @test_throws ArgumentError MultiContinuousSpace([-1, -2], [1, 2, 3])

        s = MultiContinuousSpace([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0])
        @test eltype(s) == Array{Float64,1}
        @test [0, 0, 0] in s
        @test [-1, -2, -3] in s
        @test [1, 2, 3] in s
        @test !([-2, 0, 0] in s)
        @test !([2, 2, 2] in s)

        test_samples(s)
    end

    @testset "TupleSpace and DictSpace" begin
        s = TupleSpace(
            DiscreteSpace(3),
            ContinuousSpace(0.0, 1.0),
            TupleSpace(DiscreteSpace(3), ContinuousSpace(0.0, 1.0)), # recursive
            DictSpace(
                :a => MultiDiscreteSpace([2.0, 4.0]),
                :b => TupleSpace(
                    MultiContinuousSpace([-1, -2], [2.5, 3.5]),
                    MultiDiscreteSpace([3, 2]),
                ),
            ), # combined spaces
        )

        test_samples(s)
    end

    # https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl/pull/28
    if Flux.use_cuda[]
        using CuArrays
        @testset "MultiContinuousSpace with CuArrays" begin
            low = cu([1.0 2.0; 3.0 4.0])
            high = cu([2.0 3.0; 4.0 5.0])
            s = MultiContinuousSpace(low, high)

            @test low ∈ s
            test_samples(s)
            @test high ∈ s
        end
    end
end
