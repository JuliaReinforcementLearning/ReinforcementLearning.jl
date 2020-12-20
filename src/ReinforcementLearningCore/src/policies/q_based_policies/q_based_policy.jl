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
(π::QBasedPolicy)(env, ::MinimalActionSet) = π.explorer(π.learner(env))
(π::QBasedPolicy)(env, ::FullActionSet) = π.explorer(π.learner(env), legal_action_space_mask(env))

RLBase.prob(p::QBasedPolicy, env) = prob(p, env, ActionStyle(env))
RLBase.prob(p::QBasedPolicy, env, ::MinimalActionSet) =
    prob(p.explorer, p.learner(env))
RLBase.prob(p::QBasedPolicy, env, ::FullActionSet) =
    prob(p.explorer, p.learner(env), legal_action_space_mask(env))

@forward QBasedPolicy.learner RLBase.priority

RLBase.update!(p::QBasedPolicy, trajectory::AbstractTrajectory) =
    update!(p.learner, trajectory)

function check(p::QBasedPolicy, env::AbstractEnv)
    A = action_space(env)
    if (A isa AbstractVector && A == 1:length(A)) ||
        (A isa Tuple && A == Tuple(1:length(A)))
        # this is expected
    else
        @warn "Applying a QBasedPolicy to an environment with a unknown action space. Maybe convert the environment with `discrete2standard_discrete` in ReinforcementLearningEnvironments.jl first or redesign the environment."
    end

    check(p.learner, env)
    check(p.explorer, env)
end

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
        if current_player(env) == chance_player(env)
            # this should be faster. we don't need to allocate memory to store the probability of chance node
            return rand(p.explorer.rng, action_space(env))
        end
    end
    p(env, ActionStyle(env))  # fall back to general implementation above
end

function RLBase.prob(p::TabularRandomPolicy, env, ::FullActionSet)
    m = legal_action_space_mask(env)
    prob = zeros(length(m))
    prob[m] .= p.learner(env)
    prob
end
