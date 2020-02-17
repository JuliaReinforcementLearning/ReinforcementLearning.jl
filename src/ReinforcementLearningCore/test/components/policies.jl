@testset "policies" begin

    @testset "RandomPolicy" begin
        p = RandomPolicy(DiscreteSpace(3))
        obs = (reward = 0.0, terminal = false, state = 1)

        Random.seed!(p, 321)
        actions = [p(obs) for _ in 1:100]
        Random.seed!(p, 321)
        new_actions = [p(obs) for _ in 1:100]
        @test actions == new_actions
    end

    @testset "WeightedRandomPolicy" begin
        @testset "1D" begin
            weight = [1, 2, 3]
            ratio = weight ./ sum(weight)
            N = 1000
            actions = [:a, :b, :c]
            p = WeightedRandomPolicy(weight, actions=actions, seed=123)

            samples = [p(nothing, MINIMAL_ACTION_SET) for _ in 1:N]
            stats = countmap(samples)
            for (a, r) in zip(actions, ratio)
                @test isapprox(r, stats[a]/N, atol=0.05)
            end

            legal_actions = [:aa, :cc]
            legal_actions_mask=[true, false, true]
            obs = (
                reward=0.,
                terminal=false,
                state=nothing,
                legal_actions = legal_actions,
                legal_actions_mask=legal_actions_mask,
                )
            samples = [p(obs) for _ in 1:N]
            stats = countmap(samples)

            weighted_ratio = ratio[legal_actions_mask] ./ sum(ratio[legal_actions_mask])
            for i in 1:length(legal_actions)
                @test isapprox(stats[legal_actions[i]]/N, weighted_ratio[i], atol=0.05)
            end
        end

        @testset "2D" begin
            n_state = 2
            weight = reshape(1:6, 3, n_state)
            ratio = weight ./ sum(weight, dims=1)
            N = 1000
            actions = [:a, :b, :c]

            p = WeightedRandomPolicy(weight, actions=actions, seed=123)

            for state in 1:n_state
                samples = [p((state=state,), MINIMAL_ACTION_SET) for _ in 1:N]
                stats = countmap(samples)
                for (a, r) in zip(actions, ratio[:, state])
                    @test isapprox(r, stats[a]/N, atol=0.05)
                end
            end

            for state in 1:n_state
                legal_actions = [:aa, :cc]
                legal_actions_mask=[true, false, true]
                obs = (
                    reward=0.,
                    terminal=false,
                    state=state,
                    legal_actions = legal_actions,
                    legal_actions_mask=legal_actions_mask,
                    )
                samples = [p(obs) for _ in 1:N]
                stats = countmap(samples)

                weighted_ratio = ratio[legal_actions_mask, state] ./ sum(ratio[legal_actions_mask, state])
                for i in 1:length(legal_actions)
                    @test isapprox(stats[legal_actions[i]]/N, weighted_ratio[i], atol=0.05)
                end
            end
        end
    end
end
