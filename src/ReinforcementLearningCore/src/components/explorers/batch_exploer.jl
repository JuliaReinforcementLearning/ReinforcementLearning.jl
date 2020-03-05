export BatchExplorer

struct BatchExplorer{E<:AbstractExplorer} <: AbstractExplorer
    explorer::E
end

(x::BatchExplorer)(values::AbstractMatrix) = [x.explorer(v) for v in eachcol(values)]