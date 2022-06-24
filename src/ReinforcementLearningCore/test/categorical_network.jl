using ReinforcementLearningCore, Test, Flux
@testset "CategoricalNetwork" begin
    d = CategoricalNetwork(Dense(5,3))
    s = rand(5, 10)
    a, logits = d(s, is_sampling = true)
    @test size(a) == (3,10) == size(logits)
    @test a isa Flux.OneHotMatrix
    a, logits = d(s, 4)
    @test size(a) == (3,4,10) == size(logits)
    
    #3D input
    s = rand(5,1,10)
    a, logits = d(s, is_sampling = true)
    @test size(a) == (3,1,10) == size(logits)
    @test logits isa Array{Float64, 3}
    a, logits = d(s, 4)
    @test size(a) == (3,4,10) == size(logits)
    @testset "CUDA" begin
        if CUDA.functional()
            CUDA.allowscalar(false) 
            rng = CUDA.CURAND.RNG()
            d = CategoricalNetwork(Dense(5,3) |> gpu)
            s = cu(rand(5, 10))
            a, logits = d(rng, s, is_sampling = true)
            @test size(a) == (3,10) == size(logits)
            @test a isa Flux.OneHotMatrix
            a, logits = d(rng, s, 4)
            @test size(a) == (3,4,10) == size(logits)
            
            #3D input
            s = cu(rand(5,1,10))
            a, logits = d(rng, s, is_sampling = true)
            @test size(a) == (3,1,10) == size(logits)
            a, logits = d(rng, s, 4)
            @test size(a) == (3,4,10) == size(logits)
        end
    end
end