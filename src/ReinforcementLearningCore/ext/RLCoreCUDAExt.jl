module RLCoreCUDAExt

using ReinforcementLearningCore, CUDA
import CircularArrayBuffers: CircularArrayBuffer
using LinearAlgebra

send_to_device(::CuDevice, m) = fmap(CUDA.cu, m)

function device(x::CUDA.CURAND.RNG)
    CUDA.device()
end

# Since v0.1.10 CircularArrayBuffer will adapt internal buffer into GPU
# But in RL.jl, we don't need that feature as far as I know
send_to_device(d::CuDevice, m::CircularArrayBuffer) = send_to_device(d, collect(m))

#Used for mvnormlogpdf and mvnormkldivergence
"""
    logdetLorU(LorU::AbstractMatrix)

Log-determinant of the Positive-Semi-Definite matrix A = L*U (cholesky lower and upper triangulars), given L or U. 
Has a sign uncertainty for non PSD matrices.
"""
function logdetLorU(LorU::Union{A, LowerTriangular{T, A}, UpperTriangular{T, A}}) where {T, A <: CuArray}
    return 2*sum(log.(diag(LorU)))
end

end
