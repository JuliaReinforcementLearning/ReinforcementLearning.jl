@testset "processors" begin
    @testset "StackFrames" begin
        cb = CircularArrayBuffer{Float32}(2, 3, 4)
        s = StackFrames(2, 3, 2)
        push!(cb, s)
        @test size(cb) == (2, 3, 1)

        s(ones(Float32, 2, 3))
        @test s[:, :, 1] == zeros(2, 3)
        @test s[:, :, 2] == ones(2, 3)

        push!(cb, s)
        @test size(cb) == (2, 3, 2)

        s = StackFrames(2, 3)  # one dimension lower
        s(ones(2))
        s(2 * ones(2))
        s(3 * ones(2))

        push!(cb, s)
        @test cb[:, :, end] == [1 2 3; 1 2 3]
    end
end
