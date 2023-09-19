frame_size = 84
batchsize = 256
samples_per_epoch = Int(1e6) # change this
style = SARTS
index = 1
epochs = [1]
n_epochs = length(epochs)
rng = StableRNG(123)

# TO-DO make functions to make tests modular and more widely applicable
@testset "atari_dataset_shuffle" begin
    ds = dataset(
        "pong",
        index,
        epochs;
        repo="atari-replay-datasets",
        style = style,
        rng = rng,
        is_shuffle = true,
        batchsize = batchsize
    )

    data_dict = ds.dataset
    N_samples = size(data_dict[:state])[3]

    @test N_samples == n_epochs * samples_per_epoch

    @test size(data_dict[:state]) == (frame_size, frame_size, N_samples)
    @test size(data_dict[:action]) == (N_samples,)
    @test size(data_dict[:reward]) == (N_samples,)
    @test size(data_dict[:terminal]) == (N_samples,)

    for sample in Iterators.take(ds, 3)
        @test typeof(sample) <: NamedTuple
    end

    sample1 = iterate(ds)
    sample2 = iterate(ds)

    @test sample1 != sample2

    iters = collect(Iterators.take(ds, 2))

    iter1 = iters[1]
    iter2 = iters[2]

    @test length(iters) == 2

    for iter in iters
        @test typeof(iter) <: NamedTuple{SARTS}
    end

    @test iter1 != iter2
    @test size(iter1[:state]) == (frame_size, frame_size, batchsize)
    @test size(iter1[:action]) == (batchsize,)
    @test size(iter1[:reward]) == (batchsize,)
    @test size(iter1[:terminal]) == (batchsize,)
    @test size(iter1[:next_state]) == (frame_size, frame_size, batchsize)

end

@testset "atari_dataset" begin
    ds = dataset(
        "pong",
        index,
        epochs;
        repo="atari-replay-datasets",
        style = style,
        rng = rng,
        is_shuffle = false,
        batchsize = batchsize
    )

    data_dict = ds.dataset
    N_samples = size(data_dict[:state])[3]

    @test size(data_dict[:state]) == (frame_size, frame_size, N_samples)
    @test size(data_dict[:action]) == (N_samples,)
    @test size(data_dict[:reward]) == (N_samples,)
    @test size(data_dict[:terminal]) == (N_samples,)

    for sample in Iterators.take(ds, 3)
        @test typeof(sample) <: NamedTuple
    end

    sample1 = iterate(ds)
    sample2 = iterate(ds)

    @test sample1 == sample2

    iters = collect(Iterators.take(ds, 2))

    @test length(iters) == 2

    iter1 = iters[1]
    iter2 = iters[2]

    for iter in iters
        @test typeof(iter) <: NamedTuple{SARTS}
    end

    @test iter1 != iter2

    @test size(iter1[:state]) == (frame_size, frame_size, batchsize)
    @test size(iter1[:action]) == (batchsize,)
    @test size(iter1[:reward]) == (batchsize,)
    @test size(iter1[:terminal]) == (batchsize,)
    @test size(iter1[:next_state]) == (frame_size, frame_size, batchsize)

    @test data_dict[:state][:, :, 1:batchsize] == iter1[:state]
    @test data_dict[:action][1:batchsize] == iter1[:action]
    @test data_dict[:reward][1:batchsize] == iter1[:reward]
    @test data_dict[:terminal][1:batchsize] == iter1[:terminal]
    @test data_dict[:state][:, :, 2:batchsize+1] == iter1[:next_state]

    @test data_dict[:state][:, :, batchsize+1:batchsize*2] == iter2[:state]
    @test data_dict[:action][batchsize+1:batchsize*2] == iter2[:action]
    @test data_dict[:reward][batchsize+1:batchsize*2] == iter2[:reward]
    @test data_dict[:terminal][batchsize+1:batchsize*2] == iter2[:terminal]
    @test data_dict[:state][:, :, batchsize+2:batchsize*2+1] == iter2[:next_state]
end