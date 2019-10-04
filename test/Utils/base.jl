@testset "test base" begin
    @testset "huber_loss" begin
        @testset "all linear delta" begin
            δ = 1.
            predictions = [1.5, -1.4, -1.0, 0.0]
            labels = [0.0, 1.0, 0.0, 1.5]
            expected = mean(δ .* [1.5, 2.4, 1.0, 1.5]) .- 0.5 .* δ ^2
            @test expected ≈ mean(huber_loss(labels, predictions;δ=δ))
        end
        @testset "all quadratic delta" begin
            δ = 1.
            predictions = [1.5, -1.4, -0.5, 0.0]
            labels = [1.0, -1.0, 0.0, 0.5]
            expected = mean(0.5 .* [0.5, 0.4, 0.5, 0.5] .^2)
            @test expected ≈ mean(huber_loss(labels, predictions;δ=δ))
        end
    end

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