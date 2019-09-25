@testset "episode_turn_buffer" begin
    b = episode_RTSA_buffer(; state_eltype = Int)
    @test length(b) == 0
    @test isfull(b) == false
    @test isempty(b) == true

    push!(b; reward = 0.0, terminal = true, state = 1, action = 1)

    @test length(b) == 0
    @test isfull(b) == false
    @test isempty(b) == false

    push!(b; reward = 1.0, terminal = false, state = 2, action = 2)

    @test length(b) == 1
    @test isfull(b) == false
    @test isempty(b) == false

    push!(b; reward = 2.0, terminal = true, state = 3, action = 3)

    @test length(b) == 2
    @test isfull(b) == true
    @test isempty(b) == false
    @test b[end] == (
        state = 2,
        action = 2,
        reward = 2.0,
        terminal = true,
        next_state = 3,
        next_action = 3,
    )

    push!(b; reward = 0.0, terminal = false, state = 3, action = 3)
    push!(b; reward = 3.0, terminal = false, state = 4, action = 4)
    @test b[end] == (
        state = 3,
        action = 3,
        reward = 3.0,
        terminal = false,
        next_state = 4,
        next_action = 4,
    )
    @test length(b) == 2
end