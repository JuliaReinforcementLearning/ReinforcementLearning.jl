@testset "BestResponsePolicy" begin
    env = OpenSpielEnv("kuhn_poker")
    p = TabularRandomPolicy()
    bp0 = BestResponsePolicy(p, env, 0)
    bp1 = BestResponsePolicy(p, env, 1)

    @testset "Uniform Best Response" begin
        # taken from https://github.com/deepmind/open_spiel/blob/de2c9ef57038829c2709becb0104e3b2d76725cc/open_spiel/algorithms/best_response_test.cc#L307-L315
        expected_response_0 = Dict{String,Int}(
            "0" => 1,
            "0pb" => 0,
            "1" => 1,
            "1pb" => 1,
            "2" => 0,
            "2pb" => 1,
        )
        expected_response_1 = Dict{String,Int}(
            "0b" => 0,
            "0p" => 1,
            "1b" => 1,
            "1p" => 1,
            "2b" => 1,
            "2p" => 1,
        )
        # note that for "2", the values are the same, but the lower action will be chosen since we're using `>` instead of `>=` for selecting action with a larger value

        walk(env) do x
            if RLBase.current_player(x) == 0
                @test bp0(x) == expected_response_0[state(x)]
            elseif RLBase.current_player(x) == 1
                @test bp1(x) == expected_response_1[state(x)]
            end
        end
    end

    @testset "corner case test" begin
        # https://github.com/deepmind/open_spiel/blob/de2c9ef57038829c2709becb0104e3b2d76725cc/open_spiel/algorithms/best_response_test.cc#L337
        kuhn_exploitability_descent_iter4_policy = TabularRandomPolicy(
            table = Dict(
                "0" => [0.567034158868, 0.432965841132],
                "0b" => [0.602000197743, 0.397999802257],
                "0p" => [0.520821285373, 0.479178714627],
                "0pb" => [0.621126761233, 0.378873238767],
                "1" => [0.505160629764, 0.494839370236],
                "1b" => [0.360357968472, 0.639642031528],
                "1p" => [0.520821285373, 0.479178714627],
                "1pb" => [0.378873238767, 0.621126761233],
                "2" => [0.419580194883, 0.580419805117],
                "2b" => [0.202838286881, 0.797161713119],
                "2p" => [0.5, 0.5],
                "2pb" => [0.202838286881, 0.797161713119],
            ),
        )

        bp1 = BestResponsePolicy(kuhn_exploitability_descent_iter4_policy, env, 1)
        expected_response_1 =
            Dict("0b" => 0, "0p" => 0, "1b" => 1, "1p" => 1, "2b" => 1, "2p" => 1)

        walk(env) do x
            if RLBase.current_player(x) == 1
                @test bp1(x) == expected_response_1[state(x)]
            end
        end
    end

    @testset "best response for optimal policy" begin

        bp0 = BestResponsePolicy(get_optimal_kuhn_policy(), env, 0)
        bp1 = BestResponsePolicy(get_optimal_kuhn_policy(), env, 1)
        expected_response_0 =
            Dict("0" => 0, "0pb" => 0, "1" => 0, "1pb" => 0, "2" => 0, "2pb" => 1)
        expected_response_1 =
            Dict("0b" => 0, "0p" => 0, "1p" => 0, "1b" => 0, "2p" => 1, "2b" => 1)

        walk(env) do x
            if RLBase.current_player(x) == 0
                @test bp0(x) == expected_response_0[state(x)]
            elseif RLBase.current_player(x) == 1
                @test bp1(x) == expected_response_1[state(x)]
            end
        end
    end

    @testset "check best response value" begin

        history_and_probs_0 = Dict(
            "2" => 1.5,
            "2 1 bb" => 2.0,
            "2 1 bp" => 1.0,
            "2 1 pbp" => -1.0,
            "2 1 pb" => 2.0,
            "2 1 pp" => 1.0,
            "2 0 bb" => 2.0,
            "2 1 p" => 1.5,
            "2 0 pp" => 1.0,
            "2 0 pbb" => 2.0,
            "2 0 p" => 1.5,
            "2 1 b" => 1.5,
            "2 0 bp" => 1.0,
            "2 1 pbb" => 2.0,
            "2 0" => 1.5,
            "2 1" => 1.5,
            "2 0 pb" => 2.0,
            "2 0 b" => 1.5,
            "2 0 pbp" => -1.0,
            "1 0" => 1.5,
            "0" => -0.5,
            "1 2" => -0.5,
            "0 2 p" => -1.0,
            "" => 0.5,
            "0 1" => -0.5,
            "0 2" => -0.5,
            "1 2 pp" => -1.0,
            "0 1 p" => -1.0,
            "1" => 0.5,
            "0 2 b" => -0.5,
            "1 2 pb" => -2.0,
            "0 1 b" => -0.5,
            "1 2 pbb" => -2.0,
            "0 2 bb" => -2.0,
            "1 2 b" => -0.5,
            "0 1 pb" => -1.0,
            "1 0 bp" => 1.0,
            "0 2 pp" => -1.0,
            "1 2 p" => -1.5,
            "0 2 bp" => 1.0,
            "0 1 pp" => -1.0,
            "1 0 bb" => 2.0,
            "1 2 bp" => 1.0,
            "1 0 pbp" => -1.0,
            "0 1 bp" => 1.0,
            "1 0 p" => 1.5,
            "1 2 pbp" => -1.0,
            "0 1 pbp" => -1.0,
            "1 0 pbb" => 2.0,
            "1 2 bb" => -2.0,
            "0 1 bb" => -2.0,
            "1 0 b" => 1.5,
            "0 1 pbb" => -2.0,
            "0 2 pb" => -1.0,
            "1 0 pp" => 1.0,
            "0 2 pbb" => -2.0,
            "1 0 pb" => 2.0,
            "0 2 pbp" => -1.0,
        )

        history_and_probs_1 = Dict(
            "" => 0.416666666667,
            "0" => 1.75,
            "0 1" => 1.75,
            "0 1 b" => 2.0,
            "0 1 bb" => 2.0,
            "0 1 bp" => -1.0,
            "0 1 p" => 1.5,
            "0 1 pb" => 1.5,
            "0 1 pbb" => 2.0,
            "0 1 pbp" => 1.0,
            "0 1 pp" => 1.0,
            "0 2" => 1.75,
            "0 2 b" => 2.0,
            "0 2 bb" => 2.0,
            "0 2 bp" => -1.0,
            "0 2 p" => 1.5,
            "0 2 pb" => 1.5,
            "0 2 pbb" => 2.0,
            "0 2 pbp" => 1.0,
            "0 2 pp" => 1.0,
            "1" => 0.5,
            "1 0" => -0.75,
            "1 0 b" => -1.0,
            "1 0 bb" => -2.0,
            "1 0 bp" => -1.0,
            "1 0 p" => -0.5,
            "1 0 pb" => -0.5,
            "1 0 pbb" => -2.0,
            "1 0 pbp" => 1.0,
            "1 0 pp" => -1.0,
            "1 2" => 1.75,
            "1 2 b" => 2.0,
            "1 2 bb" => 2.0,
            "1 2 bp" => -1.0,
            "1 2 p" => 1.5,
            "1 2 pb" => 1.5,
            "1 2 pbb" => 2.0,
            "1 2 pbp" => 1.0,
            "1 2 pp" => 1.0,
            "2" => -1.0,
            "2 0" => -0.75,
            "2 0 b" => -1.0,
            "2 0 bb" => -2.0,
            "2 0 bp" => -1.0,
            "2 0 p" => -0.5,
            "2 0 pb" => -0.5,
            "2 0 pbb" => -2.0,
            "2 0 pbp" => 1.0,
            "2 0 pp" => -1.0,
            "2 1" => -1.25,
            "2 1 b" => -2.0,
            "2 1 bb" => -2.0,
            "2 1 bp" => -1.0,
            "2 1 p" => -0.5,
            "2 1 pb" => -0.5,
            "2 1 pbb" => -2.0,
            "2 1 pbp" => 1.0,
            "2 1 pp" => -1.0,
        )

        p = TabularRandomPolicy()
        bp0 = BestResponsePolicy(p, env, 0)
        bp1 = BestResponsePolicy(p, env, 1)

        walk(env) do x
            @test RLZoo.best_response_value(bp0, x) ≈ history_and_probs_0[string(x.state)]
            @test RLZoo.best_response_value(bp1, x) ≈ history_and_probs_1[string(x.state)]
        end
    end

    @testset "Optimal value test" begin
        history_and_probs_0 = Dict(
            "" => -0.05555555555555558,
            "1 2 pb" => -1.0,
            "1 2 b" => -2.0,
            "0 2 pp" => -1.0,
            "0 1 bp" => 1.0,
            "2 1 bp" => 1.0,
            "2 0 pb" => 2.0,
            "1 2 pp" => -1.0,
            "2 0 b" => 1.0,
            "0 1 bb" => -2.0,
            "2 0 pp" => 1.0,
            "2 0 p" => 1.3333333333333333,
            "1 0" => 0.3333333333333333,
            "1 0 bb" => 2.0,
            "1 0 pbp" => -1.0,
            "1 2 bp" => 1.0,
            "2 0 bp" => 1.0,
            "0 1" => -1.0,
            "0 2" => -1.0,
            "1 0 pbb" => 2.0,
            "1 0 bp" => 1.0,
            "2 0 bb" => 2.0,
            "1 2 bb" => -2.0,
            "2 1" => 1.0,
            "2 1 bb" => 2.0,
            "2 0 pbp" => -1.0,
            "1 2 p" => -1.0,
            "0 2 bb" => -2.0,
            "1 0 pp" => 1.0,
            "0 2 b" => -2.0,
            "2 1 pb" => 2.0,
            "1 2 pbb" => -2.0,
            "1 2" => -1.0,
            "0 1 pb" => -1.0,
            "0 2 p" => -1.0,
            "0 2 bp" => 1.0,
            "1 0 pb" => -1.0,
            "1 2 pbp" => -1.0,
            "2 1 pp" => 1.0,
            "0 1 pp" => -1.0,
            "2 1 pbb" => 2.0,
            "2 0" => 1.3333333333333333,
            "1 0 b" => 1.0,
            "0 2 pbp" => -1.0,
            "2 0 pbb" => 2.0,
            "0 1 pbp" => -1.0,
            "0 1 b" => 0.0,
            "2 1 b" => 1.3333333333333333,
            "2 1 pbp" => -1.0,
            "2" => 1.1666666666666665,
            "1" => -0.33333333333333337,
            "0" => -1.0,
            "0 1 p" => -1.0,
            "1 0 p" => 0.3333333333333333,
            "0 2 pbb" => -2.0,
            "0 1 pbb" => -2.0,
            "2 1 p" => 1.0,
            "0 2 pb" => -1.0,
        )

        history_and_probs_1 = Dict(
            "" => 0.0555555555556,
            "0" => 0.9,
            "0 1" => 0.6,
            "0 1 b" => -1.0,
            "0 1 bb" => 2.0,
            "0 1 bp" => -1.0,
            "0 1 p" => 1.0,
            "0 1 pb" => 1.0,
            "0 1 pbb" => 2.0,
            "0 1 pbp" => 1.0,
            "0 1 pp" => 1.0,
            "0 2" => 1.2,
            "0 2 b" => 2.0,
            "0 2 bb" => 2.0,
            "0 2 bp" => -1.0,
            "0 2 p" => 1.0,
            "0 2 pb" => 1.0,
            "0 2 pbb" => 2.0,
            "0 2 pbp" => 1.0,
            "0 2 pp" => 1.0,
            "1" => 0.266666666667,
            "1 0" => -1.0,
            "1 0 b" => -1.0,
            "1 0 bb" => -2.0,
            "1 0 bp" => -1.0,
            "1 0 p" => -1.0,
            "1 0 pb" => -0.6,
            "1 0 pbb" => -2.0,
            "1 0 pbp" => 1.0,
            "1 0 pp" => -1.0,
            "1 2" => 1.53333333333,
            "1 2 b" => 2.0,
            "1 2 bb" => 2.0,
            "1 2 bp" => -1.0,
            "1 2 p" => 1.53333333333,
            "1 2 pb" => 1.53333333333,
            "1 2 pbb" => 2.0,
            "1 2 pbp" => 1.0,
            "1 2 pp" => 1.0,
            "2" => -1.0,
            "2 0" => -1.0,
            "2 0 b" => -1.0,
            "2 0 bb" => -2.0,
            "2 0 bp" => -1.0,
            "2 0 p" => -1.0,
            "2 0 pb" => -2.0,
            "2 0 pbb" => -2.0,
            "2 0 pbp" => 1.0,
            "2 0 pp" => -1.0,
            "2 1" => -1.0,
            "2 1 b" => -1.0,
            "2 1 bb" => -2.0,
            "2 1 bp" => -1.0,
            "2 1 p" => -1.0,
            "2 1 pb" => -2.0,
            "2 1 pbb" => -2.0,
            "2 1 pbp" => 1.0,
            "2 1 pp" => -1.0,
        )

        bp0 = BestResponsePolicy(get_optimal_kuhn_policy(), env, 0)
        bp1 = BestResponsePolicy(get_optimal_kuhn_policy(), env, 1)

        walk(env) do x
            @test RLZoo.best_response_value(bp0, x) ≈ history_and_probs_0[string(x.state)]
            @test RLZoo.best_response_value(bp1, x) ≈ history_and_probs_1[string(x.state)]
        end
    end
end
