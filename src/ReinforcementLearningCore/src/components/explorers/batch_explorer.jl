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

(x::BatchExplorer)(v::AbstractVector) = x.explorer(v)

Flux.testmode!(x::BatchExplorer, mode = true) = testmode!(x.explorer, mode)
