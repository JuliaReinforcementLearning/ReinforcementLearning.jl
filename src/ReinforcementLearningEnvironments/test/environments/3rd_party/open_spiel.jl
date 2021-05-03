@testset "OpenSpielEnv" begin

    for name in [
        "tic_tac_toe",
        "kuhn_poker",
        "goofspiel(imp_info=True,num_cards=4,points_order=descending)",
    ]
        @info "testing OpenSpiel: $name"
        env = OpenSpielEnv(name)
        RLBase.test_runnable!(env)
    end
end
