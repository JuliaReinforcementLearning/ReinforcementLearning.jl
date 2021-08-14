export AbstractLearner

using Flux

"""
    (learner::AbstractLearner)(env)

A learner is usually used to estimate state values, state-action values or distributional values based on experiences.
"""
abstract type AbstractLearner end

function (learner::AbstractLearner)(env) end

"""
    get_priority(p::AbstractLearner, experience)
"""
function RLBase.priority(p::AbstractLearner, experience) end

Base.show(io::IO, p::AbstractLearner) =
    AbstractTrees.print_tree(io, StructTree(p), maxdepth=get(io, :max_depth, 10))

function RLBase.update!(
    L::AbstractLearner,
    t::AbstractTrajectory,
    e::AbstractEnv,
    s::AbstractStage,
) end

function RLBase.update!(
    L::AbstractLearner,
    t::AbstractTrajectory,
    e::AbstractEnv,
    s::PreActStage,
)
    update!(L, t)
end
