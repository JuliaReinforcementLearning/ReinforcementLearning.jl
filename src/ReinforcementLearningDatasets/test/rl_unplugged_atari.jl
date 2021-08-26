@testset "rl_unplugged_atari" begin
    ds = rl_unplugged_atari_dataset(
        "Pong",
        1,
        [1, 2];
        shuffle_buffer_size = 10_000,
        tf_reader_bufsize = 1*1024*1024,
        tf_reader_sz = 10_000,
        batch_size = 256,
        n_preallocations = Threads.nthreads() * 12
    )

    @test typeof(ds)<:RingBuffer

    data_1 = take!(ds)

    frame_size = 84
    stack_size = 4

    @test size(data_1.state) == (frame_size, frame_size, stack_size, batch_size)
    @test size(data_1.next_state) == (frame_size, frame_size, stack_size, batch_size)
    @test size(data_1.action) == (batch_size,)
    @test size(data_1.next_action) == (batch_size,)
    @test size(data_1.reward) == (batch_size,)
    @test size(data_1.terminal) == (batch_size,)
    @test size(data_1.episode_id) == (batch_size,)
    @test size(data_1.episode_return) == (batch_size,)

    @test typeof(data_1.state) == Array{UInt8, 4}
    @test typeof(data_1.next_state) == Array{UInt8, 4}
    @test typeof(data_1.action) == Vector{Int64}
    @test typeof(data_1.next_action) == Vector{Int64}
    @test typeof(data_1.reward) == Vector{Float32}
    @test typeof(data_1.terminal) == Vector{Bool}
    @test typeof(data_1.episode_id) == Vector{Int64}
    @test typeof(data_1.episode_return) == Vector{Float32}

    take!(ds)
    take!(ds)
    data_2 = take!(ds)

end