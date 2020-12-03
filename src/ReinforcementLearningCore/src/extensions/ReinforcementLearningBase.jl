using CUDA
using Distributions: pdf
using Random
using Flux
using AbstractTrees

RLBase.update!(p::RandomPolicy, x) = nothing

Random.rand(s::MultiContinuousSpace{<:CuArray}) = rand(CUDA.CURAND.generator(), s)

Base.show(io::IO, p::AbstractPolicy) =
    AbstractTrees.print_tree(io, StructTree(p), get(io, :max_depth, 10))

is_expand(::AbstractEnv) = false
