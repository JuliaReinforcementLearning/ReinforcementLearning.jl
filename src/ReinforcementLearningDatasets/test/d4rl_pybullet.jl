using Base: batch_size_err_str
@testset "d4rl_pybullet" begin
    ds = dataset(
        "hopper-bullet-mixed-v0";
        repo = "d4rl-pybullet",
        style = style,
        rng = rng,
        is_shuffle = true,
        batch_size = batch_size,
    )

    n_s = 15
    n_a = 3

    N_samples = 59345

    data_dict = ds.dataset

    @test size(data_dict[:state]) == (n_s, N_samples)
    @test size(data_dict[:action]) == (n_a, N_samples)
    @test size(data_dict[:reward]) == (1, N_samples)
    @test size(data_dict[:terminal]) == (1, N_samples)

    for sample in Iterators.take(ds, 3)
        @test typeof(sample) <: NamedTuple{SARTS}
        @test size(sample[:state]) == (n_s, batch_size)
        @test size(sample[:action]) == (n_a, batch_size)
        @test size(sample[:reward]) == (1, batch_size) ||
              size(sample[:reward]) == (batch_size,)
        @test size(sample[:terminal]) == (1, batch_size) ||
              size(sample[:terminal]) == (batch_size,)
    end

end
