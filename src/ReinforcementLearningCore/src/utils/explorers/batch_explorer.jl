export BatchExplorer

using Flux

"""
    BatchExplorer(explorer::AbstractExplorer)
"""
struct BatchExplorer{E} <: AbstractExplorer
    explorer::E
end

"""
    (x::BatchExplorer)(values::AbstractMatrix)

Apply inner explorer to each column of `values`.
"""
(x::BatchExplorer)(values::AbstractMatrix) = [x.explorer(v) for v in eachcol(values)]

(x::BatchExplorer)(values::AbstractMatrix, mask::AbstractMatrix) =
    [x.explorer(v, m) for (v, m) in zip(eachcol(values), eachcol(mask))]

(x::BatchExplorer)(v::AbstractVector) = x.explorer(v)
(x::BatchExplorer)(v::AbstractVector, m::AbstractVector) = x.explorer(v, m)
