struct DiscreteSpace{S} <: AbstractDiscreteSpace end

DiscreteSpace(x) = DiscreteSpace{(x,)}()

Base.size(::DiscreteSpace{S}) where S = S
Base.eltype(::DiscreteSpace) = Int
Base.in(x::Int, ::DiscreteSpace{S}) where S = 1 <= x <= S[1]
Base.in(x, s::DiscreteSpace) = convert(Int, x) in s
Base.==(::DiscreteSpace{S1}, ::DiscreteSpace{S2}) where {S1, S2} = S1 == S2
Base.rand(rng, ::DiscreteSpace{S}) where S = rand(rng, 1:S[1])