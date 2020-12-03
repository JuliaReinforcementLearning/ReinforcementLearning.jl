export QBasedPolicy, TabularRandomPolicy

using MacroTools: @forward
using Flux
using Setfield: @set

"""
    QBasedPolicy(;learner::Q, explorer::S)

Use a Q-`learner` to generate estimations of action values.
Then an `explorer` is applied on the estimations to select an action.
"""
Base.@kwdef mutable struct QBasedPolicy{Q<:AbstractLearner,E<:AbstractExplorer} <:
                           AbstractPolicy
    learner::Q
    explorer::E
end

Flux.functor(x::QBasedPolicy) = (learner = x.learner,), y -> @set x.learner = y.learner

(π::QBasedPolicy)(env) = π(env, ActionStyle(env))
(π::QBasedPolicy)(env, ::MinimalActionSet) = get_actions(env)[π.explorer(π.learner(env))]
(π::QBasedPolicy)(env, ::FullActionSet) =
    get_actions(env)[π.explorer(π.learner(env), get_legal_actions_mask(env))]

RLBase.get_prob(p::QBasedPolicy, env) = get_prob(p, env, ActionStyle(env))
RLBase.get_prob(p::QBasedPolicy, env, ::MinimalActionSet) =
    get_prob(p.explorer, p.learner(env))
RLBase.get_prob(p::QBasedPolicy, env, ::FullActionSet) =
    get_prob(p.explorer, p.learner(env), get_legal_actions_mask(env))

@forward QBasedPolicy.learner RLBase.get_priority

RLBase.update!(p::QBasedPolicy, trajectory::AbstractTrajectory) = update!(p.learner, trajectory)

#####
# TabularRandomPolicy
#####

const TabularRandomPolicy = QBasedPolicy{<:TabularLearner,<:WeightedExplorer}

function TabularRandomPolicy(;
    rng = Random.GLOBAL_RNG,
    is_normalized = true,
    table = Dict{String,Vector{Float64}}(),
)
    QBasedPolicy(;
        learner = TabularLearner(table),
        explorer = WeightedExplorer(; is_normalized = is_normalized, rng = rng),
    )
end

function (p::TabularRandomPolicy)(env::AbstractEnv)
    if ChanceStyle(env) === EXPLICIT_STOCHASTIC
        if get_current_player(env) == get_chance_player(env)
            # this should be faster. we don't need to allocate memory to store the probability of chance node
            return rand(p.explorer.rng, get_actions(env))
        end
    end
    p(env, ActionStyle(env))  # fall back to general implementation above
end

function RLBase.get_prob(p::TabularRandomPolicy, env, ::FullActionSet)
    m = get_legal_actions_mask(env)
    prob = zeros(length(m))
    prob[m] .= p.learner(env)
    prob
end
