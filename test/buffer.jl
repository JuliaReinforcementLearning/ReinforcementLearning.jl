@testset "buffer" begin
@testset "CircularArrayBuffer" begin
    A = ones(2, 2)
    @testset "1D Int" begin
        b = CircularArrayBuffer{Int}(3)

        @test eltype(b) == Int
        @test capacity(b) == 3
        @test isfull(b) == false
        @test length(b) == 0
        @test size(b) == (0,)
        # element must has the exact same length with the element of buffer
        @test_throws DimensionMismatch push!(b, [1, 2])  

        for x in 1:3 push!(b, x) end

        @test capacity(b) == 3
        @test isfull(b) == true
        @test length(b) == 3
        @test size(b) == (3,)
        @test b[1] == 1
        @test b[end] == 3
        @test b[1:end] == [1, 2, 3]

        for x in 4:5 push!(b, x) end

        @test capacity(b) == 3
        @test length(b) == 3
        @test size(b) == (3,)
        @test b[1] == 3
        @test b[end] == 5
        @test b[1:end] == [3, 4, 5]
    end

    @testset "2D Float64" begin
        b = CircularArrayBuffer{Array{Float64,2}}(3, (2, 2))

        @test eltype(b) == Array{Float64, 2}
        @test capacity(b) == 3
        @test isfull(b) == false
        @test length(b) == 0
        @test size(b) == (2, 2, 0)
        # element must has the exact same length with the element of buffer
        @test_throws DimensionMismatch push!(b, [1., 2.])  

        for x in 1:3 push!(b, x * A) end

        @test capacity(b) == 3
        @test isfull(b) == true
        @test length(b) == 3
        @test size(b) == (2, 2, 3)
        for i in 1:3 @test b[i] == i * A end
        @test b[end] == 3 * A

        for x in 4:5 push!(b, x * ones(4)) end  # collection is also OK

        @test capacity(b) == 3
        @test length(b) == 3
        @test size(b) == (2, 2, 3)
        @test b[1] == 3 * A
        @test b[end] == 5 * A
        
        @test b[1:end] == reshape([c for x in 3:5 for c in x*A], 2, 2, 3)
    end

    @testset "getconsecutive" begin
        b = CircularArrayBuffer{Array{Float64,2}}(6, (2, 2))
        for i in 1:6 push!(b, i * A) end

        x = Array{Float64}(undef, 2, 2, 3, 2)
        x[:, :, :, 1] = reshape([c for x in 1:3 for c in x*A], 2, 2, 3)
        x[:, :, :, 2] = reshape([c for x in 4:6 for c in x*A], 2, 2, 3)

        @test getconsecutive(b, [3, 6], 3) == x
        @test_throws BoundsError getconsecutive(b, [3, 6], 4)
    end
end

@testset "CircularTurnBuffer" begin
    @testset "1D" begin
        b = CircularTurnBuffer{Int, Int, Float64, Bool}(4)
        @test eltype(b) == Turn{Int, Int, Float64, Bool}
        @test isempty(b) == true

        push!(b, Turn(1,1,1.,false,1,1))
        push!(b, Turn(2,2,2.,false,2,2))
        push!(b, Turn(3,3,3.,false,3,3))
        push!(b, Turn(4,4,4.,false,4,4))

        @test isempty(b) == false
        @test isfull(b) == true
        @test length(b) == 4
        @test getconsecutive(b, 3, 2) == Turn(
            [2, 3],
            [2, 3],
            [2., 3.],
            [false, false],
            [2, 3],
            [2, 3]
        )

        push!(b, Turn(5,5,5.,true,5,5))
        push!(b, Turn(6,6,6.,true,6,6))

        @test length(b) == 4
        @test getconsecutive(b, 3, 2) == Turn(
            [4, 5],
            [4, 5],
            [4., 5.],
            [false, true],
            [4, 5],
            [4, 5]
        )
    end
    @testset "2D" begin
        b = CircularTurnBuffer{Array{Float64, 2}, Int, Float64, Bool}(4, (2,2), (), (), ())
        t = Turn([[1. 1.];[1. 1.]], 0, 1.0, false, [[0. 0.];[0. 0.]], 0)
        push!(b, t)
        @test length(b) == 1
        @test b[1] == t
    end
end

@testset "EpisodeTurnBuffer" begin
    b = EpisodeTurnBuffer{Int, Int, Float64, Bool}()

    @test length(b) == 0

    push!(b, Turn(1,1,1.,false,1,1))
    push!(b, Turn(2,2,2.,false,2,2))
    push!(b, Turn(3,3,3.,false,3,3))
    @test getconsecutive(b, [2,3], 2) == Turn([1 2; 2 3],
                                              [1 2; 2 3],
                                              [1.0 2.0; 2.0 3.0],
                                              Bool[false false; false false],
                                              [1 2; 2 3],
                                              [1 2; 2 3])
    @test length(b) == 3

    push!(b, Turn(4,4,4.,true,4,4))
    @test isfull(b) == true
    @test length(b) == 4

    push!(b, Turn(5,5,5.,false,5,5))
    @test length(b) == 1
    empty!(b)
    @test length(b) == 0
    @test isfull(b) == false
end
end