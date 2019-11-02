@testset "CircularArrayBuffer" begin
    A = ones(2, 2)
    @testset "1D Int" begin
        b = CircularArrayBuffer{Int}(3)

        @test eltype(b) == Int
        @test capacity(b) == 3
        @test isfull(b) == false
        @test isempty(b) == true
        @test length(b) == 0
        @test size(b) == (0,)
        # element must has the exact same length with the element of buffer
        @test_throws DimensionMismatch push!(b, [1, 2])

        for x = 1:3
            push!(b, x)
        end

        @test capacity(b) == 3
        @test isfull(b) == true
        @test length(b) == 3
        @test size(b) == (3,)
        @test b[1] == 1
        @test b[end] == 3
        @test b[1:end] == [1, 2, 3]

        for x = 4:5
            push!(b, x)
        end

        @test capacity(b) == 3
        @test length(b) == 3
        @test size(b) == (3,)
        @test b[1] == 3
        @test b[end] == 5
        @test b[1:end] == [3, 4, 5]

        empty!(b)
        @test isfull(b) == false
        @test isempty(b) == true
        @test length(b) == 0
        @test size(b) == (0,)

        push!(b, 6)
        @test isfull(b) == false
        @test isempty(b) == false
        @test length(b) == 1
        @test size(b) == (1,)
        @test b[1] == 6

        push!(b, 7)
        push!(b, 8)
        @test isfull(b) == true
        @test isempty(b) == false
        @test length(b) == 3
        @test size(b) == (3,)
        @test b[[1, 2, 3]] == [6, 7, 8]

        push!(b, 9)
        @test isfull(b) == true
        @test isempty(b) == false
        @test length(b) == 3
        @test size(b) == (3,)
        @test b[[1, 2, 3]] == [7, 8, 9]
    end

    @testset "2D Float64" begin
        b = CircularArrayBuffer{Float64}(2, 2, 3)

        @test eltype(b) == Float64
        @test capacity(b) == 3
        @test isfull(b) == false
        @test length(b) == 0
        @test size(b) == (2, 2, 0)

        for x = 1:3
            push!(b, x * A)
        end

        @test capacity(b) == 3
        @test isfull(b) == true
        @test length(b) == 2 * 2 * 3
        @test size(b) == (2, 2, 3)
        for i = 1:3
            @test b[:, :, i] == i * A
        end
        @test b[:, :, end] == 3 * A

        for x = 4:5  # collection is also OK
            push!(b, x * ones(2, 2))  # collection is also OK
        end  # collection is also OK

        @test capacity(b) == 3
        @test length(b) == 2 * 2 * 3
        @test size(b) == (2, 2, 3)
        @test b[:, :, 1] == 3 * A
        @test b[:, :, end] == 5 * A

        @test b == reshape([c for x = 3:5 for c in x * A], 2, 2, 3)
    end
end
