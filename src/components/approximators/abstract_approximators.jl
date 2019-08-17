export AbstractApproximator,
       AbstractVApproximator, AbstractQApproximator,
       AbstractV, AbstractQ

abstract type AbstractApproximator end

abstract type AbstractVApproximator{T} <: AbstractApproximator end
abstract type AbstractQApproximator{T} <: AbstractApproximator end
abstract type AbstractHybridApproximator{T} <: AbstractApproximator end

const VApproximator = Union{AbstractVApproximator, AbstractHybridApproximator}
const QApproximator = Union{AbstractQApproximator, AbstractHybridApproximator}