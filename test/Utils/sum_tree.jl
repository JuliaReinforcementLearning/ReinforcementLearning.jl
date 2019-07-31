@testset "sum_tree" begin
    t = SumTree(8)

    @test capacity(t) == 8

    for i in 1:4
       push!(t, i)
    end

    @test length(t) == 4
    @test size(t) == (4,)

    for i in 5:16
        push!(t, i)
    end
    
    @test length(t) == 8
    @test size(t) == (8,)
    @test t == 9:16

    t[:] .= 1
    @test t == ones(8)
    @test all([RL.Utils.get(t, v)[1] == i for (i, v) in enumerate(0.5:1.0:8)])

    empty!(t)
    @test capacity(t) == 8
    @test length(t) == 0
end