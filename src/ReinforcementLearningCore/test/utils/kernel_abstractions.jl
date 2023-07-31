using KernelAbstractions
using KernelAbstractions: CPU

@testset "KernelAbstractions.get_backend" begin

    @test KernelAbstractions.get_backend(rand(2)) == CPU(; static=false)
    @test KernelAbstractions.get_backend(Dense(2, 3)) == CPU(; static=false)
    @test KernelAbstractions.get_backend(Conv((2, 2), 1 => 16, relu)) == CPU(; static=false)
    @test KernelAbstractions.get_backend(Chain(x -> x .^ 2, Dense(2, 3))) == CPU(; static=false)

    if (@isdefined CUDA) && CUDA.functional()
        @test KernelAbstractions.get_backend(rand(2) |> gpu) isa CUDABackend
        @test KernelAbstractions.get_backend(Dense(2, 3) |> gpu) isa CUDABackend
        @test KernelAbstractions.get_backend(Conv((2, 2), 1 => 16, relu) |> gpu) isa CUDABackend
        @test KernelAbstractions.get_backend(Chain(x -> x .^ 2, Dense(2, 3)) |> gpu) isa CUDABackend
    end

end
