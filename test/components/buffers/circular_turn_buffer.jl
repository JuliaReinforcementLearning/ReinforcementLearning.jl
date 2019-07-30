@testset "circular_turn_buffer" begin
    @testset "scalar state" begin
        b = circular_SART_buffer(
            ;capacity=2,
            state_eltype=Int,
            state_size=()
        )

        @test length(b) == 0
        @test isfull(b) == false
        @test isempty(b) == true
        @test capacity(b) == 2
        @test capacity(get_state(b)) == 3
        @test capacity(get_action(b)) == 3
        @test capacity(get_reward(b)) == 2
        @test capacity(get_terminal(b)) == 2

        push_state!(b, 1)
        push_action!(b, 1)

        @test length(b) == 0  # a turn is not finished yet
        @test isempty(b) == true  # in fact, there's already something. but to avoid confusion, let's assume it is empty for now

        push_reward!(b, 0.3)
        push_terminal!(b, true)

        @test length(b) == 1
        @test isempty(b) == false
        @test isfull(b) == false

        push_state!(b, 2)
        push_action!(b, 2)
        push_reward!(b, 0.6)
        push_terminal!(b, false)
        push_state!(b, 3)
        push_action!(b, 3)

        @test length(b) == 2
        @test isempty(b) == false
        @test isfull(b) == true

        # make sure state/action is not overwritten yet
        @test get_state(b)[1] == 1
        @test get_state(b)[end] == 3
        @test get_action(b)[1] == 1
        @test get_action(b)[end] == 3

        push_reward!(b, 0.9)
        push_terminal!(b, true)
        push_state!(b, 4)
        push_action!(b, 4)

        # make sure that old elements are removed
        @test get_state(b)[1] == 2
        @test get_state(b)[end] == 4
        @test get_action(b)[1] == 2
        @test get_action(b)[end] == 4
        @test get_reward(b)[1] == 0.6
        @test get_reward(b)[end] == 0.9
        @test get_terminal(b)[1] == false
        @test get_terminal(b)[end] == true
    end

    @testset "2d state" begin
        b = circular_SART_buffer(
            ;capacity=3,
            state_eltype=Int,
            state_size=()
        )
    end
end