@testset "circular_turn_buffer" begin
    @testset "scalar state" begin
        b = circular_RTSA_buffer(
            ;capacity=2,
            state_eltype=Int,
            state_size=()
        )

        @test length(b) == 0
        @test isfull(b) == false
        @test isempty(b) == true

        push!(b, 0.0, true, 1, 1)

        @test length(b) == 0  # a turn is not finished yet
        @test isempty(b) == false

        push!(b, 0.3, true, 2, 2)
        push!(b, 0.6, false, 3, 3)

        @test length(b) == 2
        @test isempty(b) == false
        @test isfull(b) == true

        # make sure state/action is not overwritten yet
        @test b[1] == (state=1, action=1, reward=0.3, terminal=true, next_state=2, next_action=2)
        @test b[end] == (state=2, action=2, reward=0.6, terminal=false, next_state=3, next_action=3)

        push!(b, 0.9, true, 4, 4)

        # make sure that old elements are removed
        @test b[1] == (state=2, action=2, reward=0.6, terminal=false, next_state=3, next_action=3)
        @test b[end] == (state=3, action=3, reward=0.9, terminal=true, next_state=4, next_action=4)
    end

    @testset "2d state" begin
        b = circular_RTSA_buffer(
            ;capacity=3,
            state_eltype=Array{Int, 2},
            state_size=(2, 2)
        )

        @test length(b) == 0
        @test isfull(b) == false
        @test isempty(b) == true

        push!(b, 1., false, [1 1; 1 1], 1)
        push!(b, 1., false, [2 2; 2 2], 2)

        @test length(b) == 1
        @test isfull(b) == false
        @test isempty(b) == false

        push!(b, 2., false, [3 3; 3 3], 3)
        push!(b, 3., false, [4 4; 4 4], 4)

        @test length(b) == 3
        @test isfull(b) == true
        @test isempty(b) == false
        @test b[1] == (state=[1 1; 1 1], action=1, reward=1., terminal=false, next_state=[2 2; 2 2], next_action=2)
        @test b[end] == (state=[3 3; 3 3], action=3, reward=3., terminal=false, next_state=[4 4; 4 4], next_action=4)

        push!(b, 4., false, [5 5; 5 5], 5)

        # old experience should be removed
        @test length(b) == 3
        @test isfull(b) == true
        @test b[1] == (state=[2 2; 2 2], action=2, reward=2., terminal=false, next_state=[3 3; 3 3], next_action=3)
        @test b[end] == (state=[4 4; 4 4], action=4, reward=4., terminal=false, next_state=[5 5; 5 5], next_action=5)
    end
end