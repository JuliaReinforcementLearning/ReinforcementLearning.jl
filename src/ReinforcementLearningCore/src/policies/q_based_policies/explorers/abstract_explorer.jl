export AbstractExplorer

using Flux

"""
    (p::AbstractExplorer)(x)
    (p::AbstractExplorer)(x, mask)

Define how to select an action based on action values.
"""
abstract type AbstractExplorer end

function (p::AbstractExplorer)(x) end
function (p::AbstractExplorer)(x, mask) end

"""
    prob(p::AbstractExplorer, x) -> AbstractDistribution

Get the action distribution given action values.
"""
function RLBase.prob(p::AbstractExplorer, x) end

"""
    prob(p::AbstractExplorer, x, mask)

Similar to `prob(p::AbstractExplorer, x)`, but here only the `mask`ed elements are considered.
"""
function RLBase.prob(p::AbstractExplorer, x, mask) end
