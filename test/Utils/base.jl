@testset "test base" begin

    @testset "findallmax" begin
        @test findallmax([-Inf, -Inf, -Inf]) == (-Inf, [1, 2, 3])
        @test findallmax([Inf, Inf, Inf]) == (Inf, [1, 2, 3])
        @test findallmax([0,1,2,1,2,1,0]) == (2, [3, 5])
        @test begin
            max_val, inds = findallmax([NaN, NaN, NaN])
            isnan(max_val) && inds == [1, 2, 3]
        end
    end

    @testset "discount_rewards" begin
        reward, γ = Float64[1, 2, 3], 0.5
        @test discount_rewards(reward, γ) ≈ [2.75, 3.5, 3.0]

        reward, γ = [1.0], 0.5
        @test discount_rewards(reward, γ) ≈ reward
    end
end