export Descent, InvDecay

using Flux
using Knet

import Flux.Optimise: apply!, Descent, InvDecay, Dense
import Flux: params!
import Base: show

function apply!(o::Descent, x, δ::Number)
    o.eta * δ
end

function apply!(o::InvDecay, x, δ::Number)
    γ = o.gamma
    n = get!(o.state, x, 0)
    o.state[x] = n + 1
    δ / (1 + γ * n)
end

function (a::Dense)(x)
  W, b, σ = a.W, a.b, a.σ
  σ.(W*x .+ b)
end

function Dense(in::Integer, out::Integer, σ = identity; initW = Flux.glorot_uniform, initb = n -> zeros(Float32, n), backend=:Zygote)
    if backend == :Zygote
        Dense(initW(out, in), initb(out), σ)
    elseif backend == :Knet
        Dense(Knet.param(initW(out, in)), Knet.param(initb(out)), σ)
    else
        throw(ArgumentError("unknown backend $backend"))
    end
end

params!(p::Flux.Params, x::Knet.Param, seen = IdSet()) = push!(p, x)

backend(m::T) where T = backend(T)
backend(::Type{<:Dense}) = :Zygote
backend(::Type{<:Dense{<:Any, <:Knet.Param, <:Knet.Param}}) = :Knet

backend(m::Chain) = backend(eltype(m.layers))

function Base.show(io::IO, l::Dense)
  print(io, "Dense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, "; backend=", backend(l))
  print(io, ")")
end