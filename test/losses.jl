import Flux
function testlosses()
    m = Flux.Dense(4, 4)
    x = rand(4)
    y = m(x)
    yhat = y.data .+ rand(4) .- .5
    Flux.back!(Flux.mse(yhat, y))
    Wgrad1 = deepcopy(m.W.grad)
    @. m.W.grad = 0
    Flux.back!(huberloss(yhat, y))
    @test Wgrad1 == m.W.grad
end
testlosses()
