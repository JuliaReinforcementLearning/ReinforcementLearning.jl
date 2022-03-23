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

"""
    mvnorm_kl_divergence(μ1::M, L1M::M, μ2::M, L2M::M) where M <: AbstractMatrix

GPU differentiable implementation of the kl_divergence between two MultiVariate Gaussian distributions with mean vectors `μ1, μ2` respectively and 
with cholesky decomposition of covariance matrices `L1M, L2M`.
"""
function mvnorm_kl_divergence(μ1::M, L1M::M, μ2::M, L2M::M) where M <: AbstractMatrix
    L1 = LowerTriangular(L1M)
    L2 = LowerTriangular(L2M)
    U1 = UpperTriangular(permutedims(L1M))
    U2 = UpperTriangular(permutedims(L2M))
    d = size(μ1,1)
    logdet = logdetLorU(L2M) - logdetLorU(L1M)
    M1 = L1*U1
    L2i = inv(L2)
    U2i = inv(U2)
    M2i = U2i*L2i
    X = M2i*M1
    trace = tr(X) # trace of inv(Σ2) * Σ1
    sqmahal = sum(abs2.(L2i*(μ2 .- μ1))) #mahalanobis square distance
    return (logdet - d + trace + sqmahal)/2
end

"""
    norm_kl_divergence(μ1::AbstractVecOrMat, σ1::AbstractVecOrMat, μ2::AbstractVecOrMat, σ2::AbstractVecOrMat)

GPU differentiable implementation of the kl_divergence between two MultiVariate Gaussian distributions with mean vectors `μ1, μ2` respectively and 
diagonal covariances `σ1, σ2`. Arguments must be Vectors or single-column Matrices.
"""
function norm_kl_divergence(μ1::AbstractVecOrMat, σ1::AbstractVecOrMat, μ2::AbstractVecOrMat, σ2::AbstractVecOrMat)
    d = size(μ1,1)
    logdet = sum(log.(σ2)) - sum(log.(σ1)) 
    trace = sum(σ1 ./ σ2)
    sqmahal = sum((μ2 .- μ1) .^2 ./ σ2)
    return (logdet - d + trace + sqmahal)/2
end