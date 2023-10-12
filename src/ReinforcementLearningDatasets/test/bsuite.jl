@testset "bsuite" begin
    @testset "bsuite_cartpole_shuffled" begin
        ds = rl_unplugged_bsuite_dataset(
            "cartpole",
            [1, 2],
            "full";
            is_shuffle = true,
            stochasticity = 0.0,
            shuffle_buffer_size = 10_000,
            tf_reader_bufsize = 10_000,
            tf_reader_sz = 10_000,
            batchsize = 256,
            n_preallocations = Threads.nthreads() * 12
        )

        @test typeof(ds)<:RingBuffer

        s_size = 6
        
        data_1 = take!(ds)

        @test size(data_1.state) == (s_size, batchsize)
        @test size(data_1.next_state) == (s_size, batchsize)
        @test size(data_1.action) == (batchsize,)
        @test size(data_1.reward) == (batchsize,)
        @test size(data_1.terminal) == (batchsize,)

        @test typeof(data_1.state) == Array{Float32, 2}
        @test typeof(data_1.next_state) == Array{Float32, 2}
        @test typeof(data_1.action) == Array{Int, 1}
        @test typeof(data_1.reward) == Array{Float32, 1}
        @test typeof(data_1.terminal) == Array{Bool, 1}

    end

    @testset "bsuite_cartpole" begin
        ds = rl_unplugged_bsuite_dataset(
            "cartpole",
            [1, 2],
            "full";
            is_shuffle = false,
            stochasticity = 0.0,
            shuffle_buffer_size = 10_000,
            tf_reader_bufsize = 10_000,
            tf_reader_sz = 10_000,
            batchsize = 256,
            n_preallocations = Threads.nthreads() * 12
        )

        @test typeof(ds)<:RingBuffer

        s_size = 6
        
        data_1 = take!(ds)

        @test size(data_1.state) == (s_size, batchsize)
        @test size(data_1.next_state) == (s_size, batchsize)
        @test size(data_1.action) == (batchsize,)
        @test size(data_1.reward) == (batchsize,)
        @test size(data_1.terminal) == (batchsize,)

        @test typeof(data_1.state) == Array{Float32, 2}
        @test typeof(data_1.next_state) == Array{Float32, 2}
        @test typeof(data_1.action) == Array{Int, 1}
        @test typeof(data_1.reward) == Array{Float32, 1}
        @test typeof(data_1.terminal) == Array{Bool, 1}

    end

    @testset "bsuite_catch_shuffle" begin
        ds = rl_unplugged_bsuite_dataset(
            "catch",
            [1, 2],
            "full";
            is_shuffle = true,
            stochasticity = 0.0,
            shuffle_buffer_size = 10_000,
            tf_reader_bufsize = 10_000,
            tf_reader_sz = 10_000,
            batchsize = 256,
            n_preallocations = Threads.nthreads() * 12
        )

        @test typeof(ds)<:RingBuffer

        s_size = (10, 5)
        
        data_1 = take!(ds)

        @test size(data_1.state) == (s_size[1], s_size[2], batchsize)
        @test size(data_1.next_state) == (s_size[1], s_size[2], batchsize)
        @test size(data_1.action) == (batchsize,)
        @test size(data_1.reward) == (batchsize,)
        @test size(data_1.terminal) == (batchsize,)

        @test typeof(data_1.state) == Array{Float32, 3}
        @test typeof(data_1.next_state) == Array{Float32, 3}
        @test typeof(data_1.action) == Array{Int, 1}
        @test typeof(data_1.reward) == Array{Float32, 1}
        @test typeof(data_1.terminal) == Array{Bool, 1}

    end
end