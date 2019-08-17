@testset "parameter_server" begin
    p1, p2 = param(ones(2,2)), param(zeros(2))
    param_server = ParameterServer(params(p1, p2))

    xs = ones(2, 2)
    model = Dense(param(rand(2, 2)), param(rand(2)))
    ps = params(model)

    get(param_server) do xs
        Flux.loadparams!(model, xs)
    end

    @test model.W == p1
    @test model.b == p2

    y = [0. 1.; 1. 0.]
    opt = Flux.Optimise.Descent()
    gs = Flux.gradient(() -> Flux.mse(model(xs), y), ps)
    Flux.Optimise.update!(opt, ps, gs)

    update!(param_server) do xs
        for (x, p) in zip(xs, ps)
            copyto!(Flux.data(x), Flux.data(p))
        end
    end

    @test model.W == p1
    @test model.b == p2
end