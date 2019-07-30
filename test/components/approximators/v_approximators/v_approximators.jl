@testset "v_approximators" begin
    include("aggregation.jl")
    include("fourier.jl")
    include("linear.jl")
    include("polynomial.jl")
    include("tabular.jl")
    include("tiling.jl")
end