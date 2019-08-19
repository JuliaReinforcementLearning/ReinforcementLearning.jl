export NeuralNetworkQ, update!

using Flux

"TODO: Add static size check for legal input data type."
struct NeuralNetworkQ{Tm, To, Tp}  <: AbstractQApproximator{Any}
    model::Tm
    opt::To
    ps::Tp
    function NeuralNetworkQ(model::Tm, opt::To) where {Tm, To}
        m = model
        ps = params(m)
        new{Tm, To, typeof(ps)}(m, opt, ps)
    end
end

"get Q value of some specific action"
(Q::NeuralNetworkQ)(s, a::Int) = Q(s)[a]

"get Q value of the batch"
function batch_estimate(Q::NeuralNetworkQ, s, a)
    q = Q(s)
    q[CartesianIndex.(a, axes(q, 2))]
end

"get Q value of all actions"
(Q::NeuralNetworkQ)(s) = Q.model(s)

function update!(Q::NeuralNetworkQ, loss)
    gs = Flux.gradient(() -> loss, Q.ps)
    Flux.Optimise.update!(Q.opt, Q.ps, gs)
end

function Base.copyto!(dest::NeuralNetworkQ, src::NeuralNetworkQ)
    Flux.loadparams!(dest.model, src.ps)
end