@testset "trajectories" begin

    @testset "VectorialTrajectory" begin
        b = VectorialTrajectory(; state = Vector{Int}, reward = Float64)

        @test length(b) == 0
        @test size(b) == (0,)
        @test isempty(b) == true

        t1 = (state = [1, 2], reward = 0.0)
        push!(b; t1...)

        @test length(b) == 1
        @test size(b) == (1,)
        @test b[1] == b[end] == t1
        @test isempty(b) == false
        @test get_trace(b, :state) == [t1.state]
        @test get_trace(b, :reward) == [t1.reward]

        t2 = (state = [3, 4], reward = -1)
        push!(b; t2...)

        @test length(b) == 2
        @test size(b) == (2,)
        @test b[2] == b[end] == t2
        @test isempty(b) == false
        @test get_trace(b, :state) == [t1.state, t2.state]
        @test get_trace(b, :reward) == [t1.reward, t2.reward]

        empty!(b)

        @test length(b) == 0
        @test size(b) == (0,)
        @test isempty(b) == true
    end

    @testset "VectorialCompactSARTSATrajectory" begin
        b = VectorialCompactSARTSATrajectory()

        @test length(b) == 0
        @test size(b) == (0,)
        @test isempty(b) == true

        t1 = (state = 1, action = 2)
        push!(b; t1...)
        t2 = (reward = 1.0, terminal = false, state = 2, action = 3)
        push!(b; t2...)

        @test length(b) == 1
    end

    @testset "EpisodicCompactSARTSATrajectory" begin
        b = EpisodicCompactSARTSATrajectory()

        @test length(b) == 0
        @test size(b) == (0,)
        @test isempty(b) == true

        t1 = (state = 1, action = 2)
        push!(b; t1...)

        @test length(b) == 0
        @test size(b) == (0,)
        @test isempty(b) == false
        @test get_trace(b, :state) == [1]
        @test get_trace(b, :action) == [2]
        @test get_trace(b, :reward) == []
        @test get_trace(b, :terminal) == []
        @test get_trace(b, :next_state) == []
        @test get_trace(b, :next_action) == []

        t2 = (reward = 1.0, terminal = false, state = 2, action = 3)
        push!(b; t2...)

        @test length(b) == 1
        @test size(b) == (1,)
        @test isempty(b) == false
        @test get_trace(b, :state) == [1]
        @test get_trace(b, :action) == [2]
        @test get_trace(b, :reward) == [1.0]
        @test get_trace(b, :terminal) == [false]
        @test get_trace(b, :next_state) == [2]
        @test get_trace(b, :next_action) == [3]
        @test b[1] ==
              b[end] ==
              (
                  state = 1,
                  action = 2,
                  reward = 1.0f0,
                  terminal = false,
                  next_state = 2,
                  next_action = 3,
              )

        t3 = (reward = 2.0, terminal = true, state = 3, action = 4)
        push!(b; t3...)

        @test length(b) == 2
        @test size(b) == (2,)
        @test isempty(b) == false
        @test get_trace(b, :state) == [1, 2]
        @test get_trace(b, :action) == [2, 3]
        @test get_trace(b, :reward) == [1.0, 2.0]
        @test get_trace(b, :terminal) == [false, true]
        @test get_trace(b, :next_state) == [2, 3]
        @test get_trace(b, :next_action) == [3, 4]
        @test b[end] == (
            state = 2,
            action = 3,
            reward = 2.0f0,
            terminal = true,
            next_state = 3,
            next_action = 4,
        )

        pop!(b)

        @test length(b) == 1
        @test size(b) == (1,)
        @test isempty(b) == false
        @test get_trace(b, :state) == [1]
        @test get_trace(b, :action) == [2]
        @test get_trace(b, :reward) == [1.0]
        @test get_trace(b, :terminal) == [false]
        @test get_trace(b, :next_state) == [2]
        @test get_trace(b, :next_action) == [3]
    end

    @testset "CircularCompactSARTSATrajectory" begin
        b = CircularCompactSARTSATrajectory(; capacity = 3)

        @test length(b) == 0
        @test size(b) == (0,)
        @test isempty(b) == true

        t1 = (state = 1, action = 2)
        push!(b; t1...)

        @test length(b) == 0
        @test size(b) == (0,)
        @test isempty(b) == false
        @test get_trace(b, :state) == [1]
        @test get_trace(b, :action) == [2]
        @test get_trace(b, :reward) == []
        @test get_trace(b, :terminal) == []
        @test get_trace(b, :next_state) == []
        @test get_trace(b, :next_action) == []

        t2 = (reward = 1.0, terminal = false, state = 2, action = 3)
        push!(b; t2...)

        @test length(b) == 1
        @test size(b) == (1,)
        @test isempty(b) == false
        @test get_trace(b, :state) == [1]
        @test get_trace(b, :action) == [2]
        @test get_trace(b, :reward) == [1.0]
        @test get_trace(b, :terminal) == [false]
        @test get_trace(b, :next_state) == [2]
        @test get_trace(b, :next_action) == [3]
        @test b[1] ==
              b[end] ==
              (
                  state = 1,
                  action = 2,
                  reward = 1.0f0,
                  terminal = false,
                  next_state = 2,
                  next_action = 3,
              )

        t3 = (reward = 2.0, terminal = true, state = 3, action = 4)
        push!(b; t3...)

        @test length(b) == 2
        @test size(b) == (2,)
        @test isempty(b) == false
        @test get_trace(b, :state) == [1, 2]
        @test get_trace(b, :action) == [2, 3]
        @test get_trace(b, :reward) == [1.0, 2.0]
        @test get_trace(b, :terminal) == [false, true]
        @test get_trace(b, :next_state) == [2, 3]
        @test get_trace(b, :next_action) == [3, 4]
        @test b[end] == (
            state = 2,
            action = 3,
            reward = 2.0f0,
            terminal = true,
            next_state = 3,
            next_action = 4,
        )

        pop!(b, :state, :action)
        push!(b, state = 4, action = 5)

        @test b[end] == (
            state = 2,
            action = 3,
            reward = 2.0f0,
            terminal = true,
            next_state = 4,
            next_action = 5,
        )

        pop!(b)

        @test b[1] ==
              b[end] ==
              (
                  state = 1,
                  action = 2,
                  reward = 1.0f0,
                  terminal = false,
                  next_state = 2,
                  next_action = 3,
              )
        @test length(b) == 1

        t4 = (reward = 3.0, terminal = false, state = 4, action = 5)
        push!(b; t4...)

        @test length(b) == 2
        @test size(b) == (2,)
        @test isempty(b) == false
        @test get_trace(b, :state) == [1, 2]
        @test get_trace(b, :action) == [2, 3]
        @test get_trace(b, :reward) == [1.0, 3.0]
        @test get_trace(b, :terminal) == [false, false]
        @test get_trace(b, :next_state) == [2, 4]
        @test get_trace(b, :next_action) == [3, 5]
        @test b[end] == (
            state = 2,
            action = 3,
            reward = 3.0f0,
            terminal = false,
            next_state = 4,
            next_action = 5,
        )
    end

end
