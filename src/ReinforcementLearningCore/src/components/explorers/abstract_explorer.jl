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
    get_prob(p::AbstractExplorer, x) -> AbstractDistribution

Get the action distribution given action values.
"""
function RLBase.get_prob(p::AbstractExplorer, x) end

"""
    get_prob(p::AbstractExplorer, x, mask)

Similart to `get_prob(p::AbstractExplorer, x)`, but here only the `mask`ed elements are considered.
"""
function RLBase.get_prob(p::AbstractExplorer, x, mask) end

# see discussion https://github.com/hill-a/stable-baselines/issues/819
Flux.testmode!(p::AbstractExplorer, mode = true) = @warn "trainmode/testmode will not take effect on explorer, you may consider switching to GreedyExplorer in testmode" maxlog=1
