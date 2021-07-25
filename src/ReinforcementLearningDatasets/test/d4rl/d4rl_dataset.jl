@testset "dataset_d4rl" begin
    ds, meta = dataset(
        "hopper-medium-replay-v0";
        style = SARTS,
        rng = StableRNG(123),
        is_shuffle = true,
        max_iters = 4,
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

    for sample in ds
         @test typeof(sample) <: NamedTuple #check for SARTS
    end
    sample1, state1 = iterate(ds)
    sample2, state2 = iterate(ds, state1)

    @test sample1 != sample2
end
