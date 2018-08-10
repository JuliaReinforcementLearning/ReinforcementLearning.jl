import ReinforcementLearning: preprocessstate, Box
for perdim in [true, false]
    p = StateAggregator([0, -5, 1], [3, 9, 2], [8, 10, 12], perdimension = perdim)
    if perdim
        @test preprocessstate(p, [0, -5, 1])[1] == 1.
        @test preprocessstate(p, [3, 9, 2])[end] == 1.
        @test sum(preprocessstate(p, [1, -3, 1.5])) == 3
    else
        @test preprocessstate(p, [0, -5, 1]) == 1
        @test preprocessstate(p, [3, 9, 2]) == 8*10*12
    end
end
p = RadialBasisFunctions(Box([-1, -1], [1, 1]), 20, 1.)
@test length(preprocessstate(p, randn(2))) == 20
p = RandomProjection(rand(20, 2))
@test length(preprocessstate(p, rand(2))) == 20

p0 = StateAggregator(0, 1, 5, 2)
p = TilingStateAggregator(p0, 4)
@test length(findall(x -> x != 0, preprocessstate(p, [.1, 0]))) == 4
