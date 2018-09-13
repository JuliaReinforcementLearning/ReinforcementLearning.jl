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
        b = CircularTurnBuffer{Turn{Int, Int, Float64, Bool}}(4)
        @test eltype(b) == Turn{Int, Int, Float64, Bool}
        @test isempty(b) == true
        @test isfull(b) == false
        @test length(b) == 0

        push!(b, 1,1,1.,false)
        @test_throws BoundsError b[end]
        @test length(b) == 0

        push!(b, 2,2,2.,false)

        @test b[end] == SARDSATurn(1, 1, 1., false, 2, 2)

        push!(b, 3,3,3.,false)
        push!(b, 4,4,4.,true)

        @test isempty(b) == false
        @test isfull(b) == false
        @test length(b) == 3

        push!(b, 5,5,5.,false)

        @test isempty(b) == false
        @test isfull(b) == true
        @test length(b) == 4

        push!(b, 6,6,6.,true)

        @test length(b) == 4
        @test b[end] == SARDSATurn(5, 5, 5., false, 6, 6)
    end

    @testset "2D" begin
        b = CircularTurnBuffer{Turn{Array{Float64, 2}, Int, Float64, Bool}}(4, (2,2), (), (), ())
        push!(b, [[1. 1.];[1. 1.]], 1, 1.0, false)
        push!(b, [[2. 2.];[2. 2.]], 2, 2.0, false)
        @test length(b) == 1
        @test b[1] == SARDSATurn(
            [[1. 1.];[1. 1.]],
            1, 1.0, false, [[2. 2.];[2. 2.]], 2)
    end
end

@testset "EpisodeTurnBuffer" begin
    b = EpisodeTurnBuffer{Turn{Int, Int, Float64, Bool}}()

    push!(b, 1,1,1.,false)
    push!(b, 2,2,2.,false)
    push!(b, 3,3,3.,false)
    @test length(b) == 2

    push!(b, 4,4,4.,true)
    push!(b, 5, 5)

    @test isfull(b) == true
    @test length(b) == 4

    push!(b, 6,6,6.,false)
    push!(b, 7,7,7.,false)
    @test length(b) == 1
    @test isfull(b) == false
    @test b[end] == SARDSATurn(6,6,6.,false,7,7)
end
end