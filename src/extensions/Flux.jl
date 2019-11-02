export Descent, InvDecay, backend, set_backend!

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

#####
# extend layers defined in Flux to support Knet
#####

# relief the AbstractArray constraint to support KnetArray
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

const BACKENDS_CACHE = IdDict()
const SUPPORTED_BACKENDS = Set([:Zygote, :Knet, :Any])

function set_backend!(m, x)
    if x in SUPPORTED_BACKENDS
        BACKENDS_CACHE[m] = x
    else
        @error "unknown backend $x, supported backends are $SUPPORTED_BACKENDS"
    end
end

backend(m) = haskey(BACKENDS_CACHE, m) ? BACKENDS_CACHE[m] : :Any
backend(::Dense) = :Zygote
backend(::Dense{<:Any, <:Knet.Param, <:Knet.Param}) = :Knet
backend(m::Chain) = backend(m.layers)
backend(m::Conv) = :Zygote
backend(m::Tuple{}) = :Any

function backend(m::Tuple)
    if haskey(BACKENDS_CACHE, m)
        BACKENDS_CACHE[m]
    else
        res = :Any
        res_first = backend(m[1])
        res_rest = backend(m[2:end])
        if res_first === :Any
            res = res_rest
        elseif res_rest === :Any
            res = res_first
        elseif res_first === res_rest
            res = res_first
        else
            @error "inconsistent backend detected for $m"
        end
        BACKENDS_CACHE[m] = res
        res
    end
end

function Base.show(io::IO, l::Dense)
  print(io, "Dense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, "; backend=", backend(l))
  print(io, ")")
end