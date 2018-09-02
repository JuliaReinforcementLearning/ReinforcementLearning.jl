struct MultiDiscreteSpace{N} <:AbstractSpace
    counts::Array{Int, N}
    offset::Int
end

size(s::MultiDiscreteSpace) = size(s.counts)

"To compat with Python, here we start with 0"
sample(s::MultiDiscreteSpace) = map(x -> rand(s.offset : x + s.offset -1), s.counts)
occursin(x::Int, s::MultiDiscreteSpace) = all(e -> s.offset ≤ x < e + s.offset, s.counts)
occursin(xs::Array{Int}, s::MultiDiscreteSpace) = size(s) == size(xs) &&
    all(map((e, x) -> s.offset ≤ x < e + s.offset , s.counts, xs))
==(x::MultiDiscreteSpace, y::MultiDiscreteSpace) = x.counts == y.counts && x.offset == y.offset