if !(Sys.isapple() && Sys.ARCH === :aarch64)
@testset "device" begin

    @test RLCore.device(rand(2)) == Val(:cpu)
    @test device(Dense(2, 3)) == Val(:cpu)
    @test device(Conv((2, 2), 1 => 16, relu)) == Val(:cpu)
    @test device(Chain(x -> x .^ 2, Dense(2, 3))) == Val(:cpu)

    if (@isdefined CUDA) && CUDA.functional()
        @test device(rand(2) |> gpu) isa CuDevice
        @test device(Dense(2, 3) |> gpu) isa CuDevice
        @test device(Conv((2, 2), 1 => 16, relu) |> gpu) isa CuDevice
        @test device(Chain(x -> x .^ 2, Dense(2, 3)) |> gpu) isa CuDevice
    end

end

end
