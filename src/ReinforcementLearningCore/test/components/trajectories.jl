@testset "traces" begin
    @testset "Trajectory" begin
        t = Trajectory(; state = Vector{Int}(), reward = Vector{Bool}())
        @test (:state, :reward) == keys(t)
        @test haskey(t, :state)
        @test haskey(t, :reward)
        push!(t; state = 3, reward = true)
        push!(t; state = 4, reward = false)
        @test t[:state] == [3, 4]
        @test t[:reward] == [true, false]
        pop!(t)
        @test t[:state] == [3]
        @test t[:reward] == [true]
        empty!(t)
        @test t[:state] == Int[]
        @test t[:reward] == Bool[]
    end

    @testset "CircularArraySARTTrajectory" begin
        t = CircularArraySARTTrajectory(;
            capacity = 3,
            state = Vector{Int} => (4,),
            action = Int => (),
            reward = Float32 => (),
            terminal = Bool => (),
        )

        @test length(t) == 0
        push!(t; state = ones(Int, 4), action = 1)
        @test length(t) == 0
        push!(t; reward = 1.0f0, terminal = false, state = 2 * ones(Int, 4), action = 2)
        @test length(t) == 1

        @test t[:state] == hcat(ones(Int, 4), 2 * ones(Int, 4))

        push!(t; reward = 2.0f0, terminal = false, state = 3 * ones(Int, 4), action = 3)
        @test length(t) == 2

        push!(t; reward = 3.0f0, terminal = false, state = 4 * ones(Int, 4), action = 4)
        @test length(t) == 3
        @test t[:state] == [j for i in 1:4, j in 1:4]
        @test t[:reward] == [1, 2, 3]

        # test circle works as expected
        push!(t; reward = 4.0f0, terminal = true, state = 5 * ones(Int, 4), action = 5)
        @test length(t) == 3
        @test t[:state] == [j for i in 1:4, j in 2:5]
        @test t[:reward] == [2, 3, 4]
    end

    @testset "CircularArraySLARTTrajectory" begin
        t = CircularArraySLARTTrajectory(
            capacity = 3,
            state = Vector{Int} => (4,),
            legal_actions_mask = Vector{Bool} => (4, ),
        )
        
        # test instance type is same as type
        @test isa(t, CircularArraySLARTTrajectory)

        @test length(t) == 0
        push!(t; state = ones(Int, 4), action = 1, legal_actions_mask = trues(4))
        @test length(t) == 0
        push!(t; reward = 1.0f0, terminal = false)
        @test length(t) == 1
    end

    @testset "ReservoirTrajectory" begin
        # test length
        t = ReservoirTrajectory(3; a = Array{Float64,2}, b = Bool)
        push!(t; a = rand(2, 3), b = rand(Bool))
        @test length(t) == 1
        push!(t; a = rand(2, 3), b = rand(Bool))
        @test length(t) == 2
        push!(t; a = rand(2, 3), b = rand(Bool))
        @test length(t) == 3

        for _ in 1:100
            push!(t; a = rand(2, 3), b = rand(Bool))
        end

        @test length(t) == 3

        # test distribution

        Random.seed!(110)
        k, n, N = 3, 10, 10000
        stats = Dict(i => 0 for i in 1:n)
        for _ in 1:N
            t = ReservoirTrajectory(k; a = Array{Int,2}, b = Int)
            for i in 1:n
                push!(t; a = i .* ones(Int, 2, 3), b = i)
            end

            for i in 1:length(t)
                stats[t[:b][i]] += 1
            end
        end

        for v in values(stats)
            @test isapprox(v / N, k / n; atol = 0.03)
        end
    end
end
