using KernelAbstractions: CPU, get_backend

@testset "get_backend" begin

    @test get_backend(rand(2)) == CPU(; static=false)
    @test get_backend(Dense(2, 3)) == CPU(; static=false)
    @test get_backend(Conv((2, 2), 1 => 16, relu)) == CPU(; static=false)
    @test get_backend(Chain(x -> x .^ 2, Dense(2, 3))) == CPU(; static=false)

    if CUDA.functional()
        @test get_backend(rand(2) |> gpu) isa CUDABackend
        @test get_backend(Dense(2, 3) |> gpu) isa CUDABackend
        @test get_backend(Conv((2, 2), 1 => 16, relu) |> gpu) isa CUDABackend
        @test get_backend(Chain(x -> x .^ 2, Dense(2, 3)) |> gpu) isa CUDABackend
    end

end
