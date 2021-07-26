@testset "dataset_d4rl" begin
    ds = dataset(
        "hopper-medium-replay-v0";
        style = SARTS,
        rng = StableRNG(123),
        is_shuffle = true,
        batch_size = 256
    )
    n_s = 11
    n_a = 3
    N_samples = 200919

    data_dict = ds.dataset

    @test size(data_dict[:state]) == (n_s, N_samples)
    @test size(data_dict[:action]) == (n_a, N_samples)
    @test size(data_dict[:reward]) == (N_samples,)
    @test size(data_dict[:terminal]) == (N_samples,)

    i = 1

    while i < 5
        sample = iterate(ds)
        @test typeof(sample) <: NamedTuple
        i += 1
    end

    sample1 = iterate(ds)
    sample2 = iterate(ds)

    @test sample1 != sample2
end
