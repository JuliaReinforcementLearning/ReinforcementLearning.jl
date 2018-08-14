import Flux

struct Linear{Ts}
    W::Ts
end
export Linear
function Linear(in::Int, out::Int; 
                T = Float64, initW = (out, in) -> zeros(T, out, in))
    Linear(Flux.param(initW(out, in)))
end
(a::Linear)(x) = a.W * x
@static if VERSION < v"0.7.0-beta2.199" 
    Flux.treelike(Linear)
else
    Flux.@treelike(Linear)
end

Base.show(io::IO, l::Linear) = print(io, "Linear( $(size(l.W, 2)), $(size(l.W, 1)))")

