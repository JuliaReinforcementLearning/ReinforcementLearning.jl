export DiscreteSpace
using Random:AbstractRNG

struct DiscreteSpace <: AbstractDiscreteSpace
    n::Int
    function DiscreteSpace(x::Int) 
        x > 0 || throw(ArgumentError("$x must be greater than zero"))
        new(x)
    end
end


Base.length(s::DiscreteSpace) = s.n
Base.eltype(s::DiscreteSpace) = Int
Base.in(x, s::DiscreteSpace) = 1 <= convert(Int, x) <= s.n
Base. ==(s1::DiscreteSpace, s2::DiscreteSpace) = s1.n == s2.n
Base.rand(rng::AbstractRNG, s::DiscreteSpace) = rand(rng, 1:s.n)