module RLCoreCUDAExt

using ReinforcementLearningCore, CUDA
import ReinforcementLearningCore: send_to_device, logdetLorU
import CircularArrayBuffers: CircularArrayBuffer
using LinearAlgebra
import KernelAbstractions

KernelAbstractions.get_backend(x::CUDA.CURAND.RNG) = CUDABackend()

#Used for mvnormlogpdf and mvnormkldivergence
"""
    logdetLorU(LorU::AbstractMatrix)

Log-determinant of the Positive-Semi-Definite matrix A = L*U (cholesky lower and upper triangulars), given L or U. 
Has a sign uncertainty for non PSD matrices.
"""
function logdetLorU(LorU::Union{A, LowerTriangular{T, A}, UpperTriangular{T, A}}) where {T, A <: CuArray}
    return 2*sum(log.(diag(LorU)))
end

# Since v0.1.10 CircularArrayBuffer will adapt internal buffer into GPU
# But in RL.jl, we don't need that feature as far as I know
send_to_device(d::CUDABackend, m::CircularArrayBuffer) = send_to_device(d, collect(m))

# TODO: handle multi-devices
send_to_device(::CUDABackend, m) = fmap(CUDA.cu, m)
end

