export AbstractApproximator, AbstractVApproximator, AbstractQApproximator

abstract type AbstractApproximator end

abstract type AbstractVApproximator <: AbstractApproximator end
abstract type AbstractQApproximator <: AbstractApproximator end
abstract type AbstractHybridApproximator <: AbstractApproximator end