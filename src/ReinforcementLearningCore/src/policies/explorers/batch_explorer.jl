export BatchExplorer

"""
    BatchExplorer(explorer::AbstractExplorer)
"""
struct BatchExplorer{E} <: AbstractExplorer
    explorer::E
end

"""
    RLBase.plan!(x::BatchExplorer, values::AbstractMatrix)

Apply inner explorer to each column of `values`.
"""
plan!(x::BatchExplorer, values::AbstractMatrix) = [x.explorer(v) for v in eachcol(values)]

plan!(x::BatchExplorer, values::AbstractMatrix, mask::AbstractMatrix) =
    [x.explorer(v, m) for (v, m) in zip(eachcol(values), eachcol(mask))]

plan!(x::BatchExplorer, v::AbstractVector) = x.explorer(v)
plan!(x::BatchExplorer, v::AbstractVector, m::AbstractVector) = x.explorer(v, m)
