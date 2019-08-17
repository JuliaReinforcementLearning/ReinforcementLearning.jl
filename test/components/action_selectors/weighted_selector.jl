@testset "WeightedSelector" begin
    Random.seed!(123)
    s = WeightedSelector(true)
    values = [0.2, 0.3, 0.5]
    action_counts = countmap([s(values; step=i) for i in 1:1000])
    for i in 1:length(values)
        @test isapprox(action_counts[i]/1000, values[i], atol=0.1)
    end
end