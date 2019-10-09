import Base:copyto!
import Zygote:gradient

using Knet
using Zygote

Base.copyto!(dest::Knet.Param, src::Knet.Param) = copyto!(value(dest), value(src))

function gradient(f, ::Val{:Knet}, ps::Zygote.Params)
    gs = @diff f()
    Zygote.Grads(IdDict(p => grad(gs, p) for p in ps))
end