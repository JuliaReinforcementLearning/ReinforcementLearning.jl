struct DiscreteSpace <: AbstractSpace
    n::Int
    offset::Int
end

size(d::DiscreteSpace) = 1
sample(d::DiscreteSpace) = rand(d.offset : d.n + d.offset - 1)
occursin(x::Int, d::DiscreteSpace) = d.offset ≤ x < d.offset + d.n
==(x::DiscreteSpace, y::DiscreteSpace) = x.n == y.n && x.offset == y.offset