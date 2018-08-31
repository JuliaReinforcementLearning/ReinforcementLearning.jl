struct MultiBinarySpace <: AbstractSpace
    size::Tuple{Vararg{Int}}
end

MultiBinarySpace(sz::Vararg{Int}) = MultiBinarySpace(sz)

size(s::MultiBinarySpace) = s.size

sample(s::MultiBinarySpace) = rand(Bool, s.size...)
occursin(x::Array{Bool}, s::MultiBinarySpace) = size(s) == size(x)
==(x::MultiBinarySpace, y::MultiBinarySpace) = x.size == y.size