@testset "test base" begin

    @testset "findallmax" begin
        @test findallmax([-Inf, -Inf, -Inf]) == (-Inf, [1, 2, 3])
        @test findallmax([Inf, Inf, Inf]) == (Inf, [1, 2, 3])
        @test findallmax([0, 1, 2, 1, 2, 1, 0]) == (2, [3, 5])
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

    @testset "SampleAvg" begin
        f = SampleAvg()
        @test f(2) ≈ 2.0
        @test f(3) ≈ (2 + 3) / 2
        @test f(5) ≈ (2 + 3 + 5) / 3
    end

    @testset "CachedSampleAvg" begin
        f = CachedSampleAvg()
        @test f(:a, 3) ≈ 3
        @test f(:a, 5) ≈ (3 + 5) / 2
        @test f(:a, 8) ≈ (3 + 5 + 8) / 3
        @test f(:b, 0) ≈ 0
    end

    @testset "CachedSum" begin
        f = CachedSum()
        @test f(:a, 1) ≈ 1
        @test f(:a, 2) ≈ 3
        @test f(:a, 3) ≈ 6
        @test f(:b, 0.3) ≈ 0.3
    end
end