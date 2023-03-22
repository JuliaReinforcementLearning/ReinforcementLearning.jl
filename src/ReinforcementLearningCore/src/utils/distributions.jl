export normlogpdf, mvnormlogpdf, diagnormlogpdf, mvnormkldivergence, diagnormkldivergence, normkldivergence

using Flux: unsqueeze, stack
using LinearAlgebra

# watch https://github.com/JuliaStats/Distributions.jl/issues/1183
const log2π = log(2.0f0π)

"""
     normlogpdf(μ, σ, x; ϵ = 1.0f-8)

GPU automatic differentiable version for the logpdf function of a univariate normal distribution.
Adding an epsilon value to guarantee numeric stability if sigma is exactly zero
(e.g. if relu is used in output layer).
"""
function normlogpdf(μ, σ, x; ϵ=1.0f-8)
    z = (x .- μ) ./ (σ .+ ϵ)
    -(z .^ 2 .+ log2π) / 2.0f0 .- log.(σ .+ ϵ)
end

"""
    diagnormlogpdf(μ, σ, x; ϵ = 1.0f-8)

GPU automatic differentiable version for the logpdf function of normal distributions with 
diagonal covariance. Adding an epsilon value to guarantee numeric stability if sigma is 
exactly zero (e.g. if relu is used in output layer).
"""
function diagnormlogpdf(μ, σ, x; ϵ = 1.0f-8)
    v = (σ .+ ϵ) .^2
    -0.5f0*(log(prod(v)) .+ inv.(v)'*((x .- μ).^2) .+ length(μ)*log2π)
end

#3D tensor version
function diagnormlogpdf(μ::AbstractArray{<:Any,3}, σ::AbstractArray{<:Any,3}, x::AbstractArray{<:Any,3}; ϵ = 1.0f-8)
    logp = [diagnormlogpdf(μ[:, :, k], σ[:, :, k], x[:, :, k]) for k in 1:size(x, 3)]
    return reduce((x,y)->cat(x,y,dims=3), logp) #returns a 3D vector 
end

"""
    mvnormlogpdf(μ::AbstractVecOrMat, L::AbstractMatrix, x::AbstractVecOrMat)

GPU automatic differentiable version for the logpdf function of multivariate
normal distributions.  Takes as inputs `mu` the mean vector, `L` the lower
triangular matrix of the cholesky decomposition of the covariance matrix, and
`x` a matrix of samples where each column is a sample.  Return a Vector
containing the logpdf of each column of x for the `MvNormal` parametrized by `μ`
and `Σ = L*L'`.
"""
function mvnormlogpdf(μ::AbstractVecOrMat, L::AbstractMatrix, x::AbstractVecOrMat)
    return -(
        (size(x, 1) * log2π + logdetLorU(L)) .+ vec(sum(abs2.(L \ (x .- μ)), dims=1))
    ) ./ 2
end


"""
    mvnormlogpdf(μ::A, LorU::A, x::A; ϵ = 1f-8) where A <: AbstractArray

Batch version that takes 3D tensors as input where each slice along the 3rd
dimension is a batch sample.  `μ` is a (action_size x 1 x batch_size) matrix,
`L` is a (action_size x action_size x batch_size), x is a (action_size x
action_samples x batch_size).  Return a 3D matrix of size (1 x action_samples x
batch_size). 
"""
function mvnormlogpdf(μ::A, LorU::A, x::A; ϵ=1.0f-8) where {A<:AbstractArray}
    logp = [mvnormlogpdf(μs, LorUs, xs) for (μs, LorUs, xs) in zip(eachslice(μ, dims = 3), eachslice(LorU, dims = 3), eachslice(x, dims = 3))]
    return unsqueeze(stack(logp; dims=2), dims=1) #returns a 3D vector 
end

#Used for mvnormlogpdf and mvnormkldivergence
"""
    logdetLorU(LorU::AbstractMatrix)

Log-determinant of the Positive-Semi-Definite matrix A = L*U (cholesky lower and upper triangulars), given L or U. 
Has a sign uncertainty for non PSD matrices.
"""
function logdetLorU(LorU::Union{A, LowerTriangular{T, A}, UpperTriangular{T, A}}) where {T, A <: CuArray}
    return 2*sum(log.(diag(LorU)))
end

#Cpu fallback
logdetLorU(LorU::AbstractMatrix) = logdet(LorU)*2

"""	
    mvnormkldivergence(μ1, L1, μ2, L2)
    
GPU differentiable implementation of the kl_divergence between two MultiVariate Gaussian distributions with mean vectors `μ1, μ2` respectively and 	
with cholesky decomposition of covariance matrices `L1, L2`.	
"""	
function mvnormkldivergence(μ1, L1M, μ2, L2M)
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
    diagnormkldivergence(μ1, σ1, μ2, σ2)	

GPU differentiable implementation of the kl_divergence between two MultiVariate Gaussian distributions with mean vectors `μ1, μ2` respectively and 	
diagonal standard deviations `σ1, σ2`. Arguments must be Vectors or single-column Matrices.	
"""	
function diagnormkldivergence(μ1, σ1, μ2, σ2)	
    v1, v2 = σ1.^2, σ2.^2
    d = size(μ1,1)	
    logdet = sum(log.(v2)) - sum(log.(v1)) 	
    trace = sum(v1 ./ v2)	
    sqmahal = sum((μ2 .- μ1) .^2 ./ v2)	
    return (logdet - d + trace + sqmahal)/2	
end

"""	
    normkldivergence(μ1, σ1, μ2, σ2)	

GPU differentiable implementation of the kl_divergence between two univariate Gaussian 
distributions with means `μ1, μ2` and standard deviations `σ1, σ2` respectively.	
"""	
function normkldivergence(μ1, σ1, μ2, σ2)	
    log(σ2) - log(σ1) + (σ1^2 + (μ1 - μ2)^2)/(2σ2^2) - typeof(μ1)(0.5)
end
