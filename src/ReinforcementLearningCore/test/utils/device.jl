@testset "device" begin

    @test device(rand(2)) == Val(:cpu)
    @test device(Dense(2, 3)) == Val(:cpu)
    @test device(Conv((2, 2), 1 => 16, relu)) == Val(:cpu)
    @test device(Chain(x -> x .^ 2, Dense(2, 3))) == Val(:cpu)

    if CUDA.functional()
        @test device(rand(2) |> gpu) == Val(:gpu)
        @test device(Dense(2, 3) |> gpu) == Val(:gpu)
        @test device(Conv((2, 2), 1 => 16, relu) |> gpu) == Val(:gpu)
        @test device(Chain(x -> x .^ 2, Dense(2, 3)) |> gpu) == Val(:gpu)
    end

end
