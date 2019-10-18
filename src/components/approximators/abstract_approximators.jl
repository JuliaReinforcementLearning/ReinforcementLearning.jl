export AbstractApproximator, AbstractVApproximator, AbstractQApproximator

"""
An approximator is a functional object to estimate a state.
Two typical kinds of approximators are
[`AbstractVApproximator`](@ref) and [`AbstractQApproximator`](@ref).
"""
abstract type AbstractApproximator end

"""
A collection of approximators to estimate the value of a state.
"""
abstract type AbstractVApproximator <: AbstractApproximator end

"""
A collection of approximators to estimate the values of actions given a state.
"""
abstract type AbstractQApproximator <: AbstractApproximator end