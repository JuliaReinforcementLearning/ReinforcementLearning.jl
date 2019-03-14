using StaticArrays

struct MultiDiscreteSpace{S, T, N, L} <:AbstractSpace
    counts::SArray{S, T, N, L}
end

MultiDiscreteSpace(x) = MultiDiscreteSpace(@SArray(x))

Base.size(::MultiDiscreteSpace{S, T, N, L}) where {S, T, N, L} = S
Base.eltype(s::MultiDiscreteSpace{S, T, N, L}) where {S, T, N, L} = SArray{S, T, N, L}
Base.in(x, s::MultiDiscreteSpace{S, T, N, L}) where {S, T, N, L} = length(x) == 
Base.==(s1::DiscreteSpace, s2::DiscreteSpace) = s1.n == s2.n
Base.rand(rng, s::DiscreteSpace) = rand(rng, 1:s.n)