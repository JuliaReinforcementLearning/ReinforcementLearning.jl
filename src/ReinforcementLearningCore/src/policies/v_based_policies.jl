export VBasedPolicy

function default_value_action_mapping(env, value_learner; explorer = GreedyExplorer())
    A = legal_action_space(env)
    V = map(A) do a
        value_learner(child(env, a))
    end
    A[explorer(V)]
end

"""
    VBasedPolicy(;learner, mapping=default_value_action_mapping)

The `learner` must be a value learner. The `mapping` is a function which returns
an action given `env` and the `learner`. By default we iterate through all the
valid actions and select the best one which lead to the maximum state value.
"""
Base.@kwdef struct VBasedPolicy{L,M} <: AbstractPolicy
    learner::L
    mapping::M = default_value_action_mapping
end

(p::VBasedPolicy)(env::AbstractEnv) = p.mapping(env, p.learner)

function RLBase.update!(
    p::VBasedPolicy,
    t::AbstractTrajectory,
    e::AbstractEnv,
    s::AbstractStage,
)
    update!(p.learner, t, e, s)
end
