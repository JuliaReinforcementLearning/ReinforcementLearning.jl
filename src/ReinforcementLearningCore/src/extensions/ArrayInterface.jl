using ArrayInterface

function ArrayInterface.restructure(x::AbstractArray{T1, 0}, y::AbstractArray{T2, 0}) where {T1, T2}
    out = similar(x, eltype(y))
    out .= y
    out
end