export AbstractActionSelector, AbstractDiscreteActionSelector

"""
    AbstractActionSelector

Take in an estimation and return an action.
"""
abstract type AbstractActionSelector end

"""
    AbstractDiscreteActionSelector

Generate an action given the estimated value of different actions.

| Required Methods| Brief Description |
|:----------------|:------------------|
| `selector(values; kwargs...)` | `selector`, an instance of `AbstractDiscreteActionSelector`, must be a callable object which takes in an estimation of all discrete actions and returns an action. |
"""
abstract type AbstractDiscreteActionSelector <: AbstractActionSelector end