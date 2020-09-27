@testset "ReservoirArrayBuffer" begin
    b = ReservoirArrayBuffer{Int}(3, 2)
    @assert size(b) == (3, 0)

    push!(b, [1, 1, 1])
    @assert size(b) == (3, 1)
    @test all(b .== [1; 1; 1])

    push!(b, [2, 2, 2])
    @assert size(b) == (3, 2)
    @test all(b .== [1 2; 1 2; 1 2])

    push!(b, [0, 0, 0])

    @test size(b) == (3, 2)
end
