export BatchExplorer

using Flux

"""
    BatchExplorer(explorer::AbstractExplorer)
"""
struct BatchExplorer{E} <: AbstractExplorer
    explorer::E
end

BatchExplorer(explorers::AbstractExplorer...) = BatchExplorer(explorers)

"""
    (x::BatchExplorer)(values::AbstractMatrix)

Apply inner explorer to each column of `values`.
"""
(x::BatchExplorer)(values::AbstractMatrix) = [x.explorer(v) for v in eachcol(values)]

(x::BatchExplorer{<:Tuple})(values::AbstractMatrix) =
    [explorer(v) for (explorer, v) in zip(x.explorer, eachcol(values))]

Flux.testmode!(x::BatchExplorer, mode=true) = testmode!(x.explorer, mode)

function Flux.testmode!(x::BatchExplorer{<:Tuple}, mode=true)
    for p in x.explorer
        testmode!(p, mode)
    end
end