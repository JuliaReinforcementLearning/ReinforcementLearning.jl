@testset "episode_turn_buffer" begin
    b = episode_RTSA_buffer(;state_eltype=Int)
    @test length(b) == 0
    @test isfull(b) == false
    @test isempty(b) == true

    push!(b, 0., true, 1, 1)

    @test length(b) == 0
    @test isfull(b) == false
    @test isempty(b) == false

    push!(b, 1., false, 2, 2)

    @test length(b) == 1
    @test isfull(b) == false
    @test isempty(b) == false

    push!(b, 2., true, 3, 3)

    @test length(b) == 2
    @test isfull(b) == true
    @test isempty(b) == false
    @test b[end] == (state=2, action=2, reward=2., terminal=true, next_state=3, next_action=3)

    empty!(b)

    push!(b, 0., false, 3, 3)
    push!(b, 3., false, 4, 4)
    @test b[end] == (state=3, action=3, reward=3., terminal=false, next_state=4, next_action=4)
end