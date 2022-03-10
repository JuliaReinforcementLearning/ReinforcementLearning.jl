export normlogpdf, mvnormlogpdf

using Distributions: DiscreteNonParametric, support, probs
using Flux, LinearAlgebra
# watch https://github.com/JuliaStats/Distributions.jl/issues/1183
const log2π = log(2f0π)
"""
     normlogpdf(μ, σ, x; ϵ = 1.0f-8)
GPU automatic differentiable version for the logpdf function of normal distributions.
Adding an epsilon value to guarantee numeric stability if sigma is exactly zero
(e.g. if relu is used in output layer).
"""
function normlogpdf(μ, σ, x; ϵ = 1.0f-8)
    z = (x .- μ) ./ (σ .+ ϵ)
    -(z .^ 2 .+ log2π) / 2.0f0 .- log.(σ .+ ϵ)
end

"""
    mvnormlogpdf(μ::AbstractVecOrMat, L::AbstractMatrix, x::AbstractVecOrMat)
GPU automatic differentiable version for the logpdf function of multivariate normal distributions. 
Takes as inputs `mu` the mean vector, `L` the lower triangular matrix of the cholesky decomposition of the covariance matrix, and `x` a matrix of samples where each column is a sample.
Return a Vector containing the logpdf of each column of x for the `MvNormal` parametrized by `μ` and `Σ = L*L'`.
"""
function mvnormlogpdf(μ::AbstractVecOrMat, L::AbstractMatrix, x::AbstractVecOrMat)
    return -((size(x, 1) * log2π + logdetLorU(L)) .+ vec(sum(abs2.(L\(x .- μ)), dims=1))) ./ 2
end


"""
    mvnormlogpdf(μ::A, LorU::A, x::A; ϵ = 1f-8) where A <: AbstractArray
Batch version that takes 3D tensors as input where each slice along the 3rd dimension is a batch sample.
`μ` is a (action_size x 1 x batch_size) matrix, `L` is a (action_size x action_size x batch_size), x is a (action_size x action_samples x batch_size).
Return a 3D matrix of size (1 x action_samples x batch_size). 
"""
function mvnormlogpdf(μ::A, LorU::A, x::A; ϵ = 1f-8) where A <: AbstractArray 
    logp = [mvnormlogpdf(μ[:,:,k], LorU[:,:,k], x[:,:,k]) for k in 1:size(x, 3)]
    return Flux.unsqueeze(Flux.stack(logp, 2),1) #returns a 3D vector 
end