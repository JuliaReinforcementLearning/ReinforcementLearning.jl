export normlogpdf, mvnormlogpdf, diagnormlogpdf, mvnormkldivergence, diagnormkldivergence, normkldivergence

using Flux: unsqueeze
using LinearAlgebra

using GPUArrays

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

GPU compatible and automatically differentiable version for the logpdf function of normal distributions with 
diagonal covariance. Adding an epsilon value to guarantee numeric stability if sigma is 
exactly zero (e.g. if relu is used in output layer). Accepts arguments of the same shape:
vectors, matrices or 3D array (with dimension 2 of size 1).
"""
function diagnormlogpdf(μ::AbstractArray, σ::AbstractArray, x::AbstractArray; ϵ = 1.0f-8)
    v = (σ .+ ϵ) .^2
    -0.5f0 .* (log.(prod(v, dims = 1)) .+ sum(((x .- μ).^2)./v, dims = 1) .+ size(μ, 1)*log2π)
end

"""
    mvnormlogpdf(μ::AbstractVecOrMat, L::AbstractMatrix, x::AbstractVecOrMat)

GPU compatible and automatically differentiable version for the logpdf function of multivariate
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
dimension is a batch sample.  `μ` is a (action_size x 1 x batchsize) matrix,
`L` is a (action_size x action_size x batchsize), x is a (action_size x
action_samples x batchsize).  Return a 3D matrix of size (1 x action_samples x
batchsize). 
"""
function mvnormlogpdf(μ::A, LorU::A, x::A; ϵ=1.0f-8) where {A<:AbstractArray}
    it = zip(eachslice(μ, dims = 3), eachslice(LorU, dims = 3), eachslice(x, dims = 3))
    logp = [mvnormlogpdf(μs, LorUs, xs) for (μs, LorUs, xs) in it]
    return unsqueeze(stack(logp; dims=2), dims=1)
end

#Used for mvnormlogpdf and mvnormkldivergence
"""
    logdetLorU(LorU::AbstractMatrix)

Log-determinant of the Positive-Semi-Definite matrix A = L*U (cholesky lower and upper triangulars), given L or U. 
Has a sign uncertainty for non PSD matrices.
"""
function logdetLorU(LorU::Union{A, LowerTriangular{T, A}, UpperTriangular{T, A}}) where {T, A <: AbstractGPUArray}
    return 2*sum(log.(diag(LorU)))
end

#Cpu fallback
logdetLorU(LorU::AbstractMatrix) = logdet(LorU)*2

"""	
    mvnormkldivergence(μ1, L1, μ2, L2)
    
GPU differentiable implementation of the kl_divergence between two MultiVariate Gaussian distributions with mean vectors `μ1, μ2` respectively and 	
with cholesky decomposition of covariance matrices `L1, L2`.	
"""	
function mvnormkldivergence(μ1::AbstractVecOrMat, L1M::AbstractMatrix, μ2::AbstractVecOrMat, L2M::AbstractMatrix)
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
    trace = sum(diag(X)) # trace of inv(Σ2) * Σ1	
    sqmahal = sum(abs2.(L2i*(μ2 .- μ1))) #mahalanobis square distance	
    return (logdet - d + trace + sqmahal)/2	
end	

function mvnormkldivergence(μ1::AbstractArray{T, 3}, L1::AbstractArray{T, 3}, μ2::AbstractArray{T, 3}, L2::AbstractArray{T, 3}) where T <: Real
    it = zip(eachslice(μ1, dims = 3), eachslice(L1, dims = 3), eachslice(μ2, dims = 3), eachslice(L2, dims = 3))
    kldivs = [mvnormkldivergence(m1,l1,m2,l2) for (m1,l1,m2,l2) in it]
    return reshape(kldivs, :, 1, length(kldivs))
end

"""	
    diagnormkldivergence(μ1, σ1, μ2, σ2)	

GPU differentiable implementation of the kl_divergence between two MultiVariate Gaussian distributions with mean vectors `μ1, μ2` respectively and 	
diagonal standard deviations `σ1, σ2`. Arguments must be Vectors or arrays of column vectors.	
"""	
function diagnormkldivergence(μ1::T, σ1::T, μ2::T, σ2::T) where T <: AbstractVecOrMat	
    v1, v2 = σ1.^2, σ2.^2
    d = size(μ1,1)	
    logdet = sum(log.(v2), dims = 1) - sum(log.(v1), dims = 1) 	
    trace = sum(v1 ./ v2, dims = 1)	
    sqmahal = sum((μ2 .- μ1) .^2 ./ v2, dims = 1)	
    return (logdet .- d .+ trace .+ sqmahal) ./ 2	
end

function diagnormkldivergence(μ1::T, σ1::T, μ2::T, σ2::T) where T <: AbstractArray{<: Real, 3}
    divs = diagnormkldivergence(dropdims(μ1, dims = 2), dropdims(σ1, dims = 2), dropdims(μ2, dims = 2), dropdims(σ2, dims = 2))
    return unsqueeze(divs, dims = 2)
end

"""	
    normkldivergence(μ1, σ1, μ2, σ2)	

GPU differentiable implementation of the kl_divergence between two univariate Gaussian 
distributions with means `μ1, μ2` and standard deviations `σ1, σ2` respectively.	
"""	
function normkldivergence(μ1, σ1, μ2, σ2)	
    log(σ2) - log(σ1) + (σ1^2 + (μ1 - μ2)^2)/(2σ2^2) - typeof(μ1)(0.5)
end
