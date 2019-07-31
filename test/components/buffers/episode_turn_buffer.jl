@testset "episode_turn_buffer" begin
    b = episode_SART_buffer(;state_eltype=Int)
    @test length(b) == 0
    @test isfull(b) == false
    @test isempty(b) == true
    @test capacity(b) == typemax(Int)

    push_state!(b, 1)
    push_action!(b, 1)

    push_reward!(b, 1.)
    push_terminal!(b, false)
    push_state!(b, 2)
    push_action!(b, 2)

    @test length(b) == 1
    @test isfull(b) == false
    @test isempty(b) == false

    push_reward!(b, 2.)
    push_terminal!(b, true)
    push_state!(b, 3)
    push_action!(b, 3)

    @test length(b) == 2
    @test isfull(b) == true
    @test isempty(b) == false
    @test b[end] == (state=2, action=2, reward=2., terminal=true)

    empty!(b)

    push_state!(b, 3)
    push_action!(b, 3)
    push_reward!(b, 3.)
    push_terminal!(b, false)
    @test b[end] == (state=3, action=3, reward=3., terminal=false)
end