export AbstractDiscreteActionSelector

abstract type AbstractActionSelector end

"""
    AbstractDiscreteActionSelector

A subtype of `AbstractDiscreteActionSelector` is used to generate an action
given the estimated value of different actions.

| Required Methods| Brief Description |
|:----------------|:------------------|
| `selector(values;step, kw...)` | `selector`, an instance of `AbstractDiscreteActionSelector`, must be a callable object which takes in an estimation and returns an action. |
"""
abstract type AbstractDiscreteActionSelector <: AbstractActionSelector end