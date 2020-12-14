@testset "Test convert between CommonRLEnv and RLBase" begin

    x = MontyHallEnv(;rng=MersenneTwister(111))
    y = convert(CRL.AbstractEnv, x) |> CRL.clone
    z = convert(AbstractEnv, y) |> copy

    @test ActionStyle(z) === ActionStyle(x)
    @test CRL.provided(CRL.valid_actions, y) == true
    @test CRL.provided(CRL.valid_action_mask, y) == true

    @test state(x) == CRL.observe(y) == state(z)
    @test action_space(x) == CRL.actions(y) == action_space(z)
    @test legal_action_space(x) == CRL.valid_actions(y) == legal_action_space(z)
    @test legal_action_space_mask(x) == CRL.valid_action_mask(y) == legal_action_space_mask(z)
    @test current_player(x) == CRL.player(y) == current_player(z)
    @test is_terminated(x) == CRL.terminated(y) == is_terminated(z)

    a = rand(action_space(x))
    x(a)
    r = CRL.act!(y, a)
    z(a)

    @test state(x) == CRL.observe(y) == state(z)
    @test action_space(x) == CRL.actions(y) == action_space(z)
    @test legal_action_space(x) == CRL.valid_actions(y) == legal_action_space(z)
    @test legal_action_space_mask(x) == CRL.valid_action_mask(y) == legal_action_space_mask(z)
    @test current_player(x) == CRL.player(y) == current_player(z)
    @test is_terminated(x) == CRL.terminated(y) == is_terminated(z)

    reset!(x)
    CRL.reset!(y)
    reset!(z)

    @test state(x) == CRL.observe(y) == state(z)
    @test action_space(x) == CRL.actions(y) == action_space(z)
    @test legal_action_space(x) == CRL.valid_actions(y) == legal_action_space(z)
    @test legal_action_space_mask(x) == CRL.valid_action_mask(y) == legal_action_space_mask(z)
    @test current_player(x) == CRL.player(y) == current_player(z)
    @test is_terminated(x) == CRL.terminated(y) == is_terminated(z)
end
