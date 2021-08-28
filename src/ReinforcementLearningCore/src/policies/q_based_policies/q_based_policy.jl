export QBasedPolicy, TabularRandomPolicy

using MacroTools: @forward
using Flux
using Distributions: Distribution, probs
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

(π::QBasedPolicy)(env) = π(env, ActionStyle(env), action_space(env))

(π::QBasedPolicy)(env, ::MinimalActionSet, ::Base.OneTo) = π.explorer(π.learner(env))
(π::QBasedPolicy)(env, ::FullActionSet, ::Base.OneTo) =
    π.explorer(π.learner(env), legal_action_space_mask(env))

(π::QBasedPolicy)(env, ::MinimalActionSet, A) = A[π.explorer(π.learner(env))]
(π::QBasedPolicy)(env, ::FullActionSet, A) =
    A[π.explorer(π.learner(env), legal_action_space_mask(env))]

RLBase.prob(p::QBasedPolicy, env::AbstractEnv) = prob(p, env, ActionStyle(env))
RLBase.prob(p::QBasedPolicy, env::AbstractEnv, ::MinimalActionSet) =
    prob(p.explorer, p.learner(env))
RLBase.prob(p::QBasedPolicy, env::AbstractEnv, ::FullActionSet) =
    prob(p.explorer, p.learner(env), legal_action_space_mask(env))

function RLBase.prob(p::QBasedPolicy, env::AbstractEnv, action)
    A = action_space(env)
    P = prob(p, env)
    if P isa Distribution
        P = probs(P)
    end
    @assert length(A) == length(P)
    if A isa Base.OneTo
        P[action]
        # elseif A isa ZeroTo
        #     P[action+1]
    else
        for (a, p) in zip(A, P)
            if a == action
                return p
            end
        end
        @error "action[$action] is not found in action space[$(action_space(env))]"
    end
end

@forward QBasedPolicy.learner RLBase.priority

function RLBase.update!(
    p::QBasedPolicy,
    t::AbstractTrajectory,
    e::AbstractEnv,
    s::AbstractStage,
)
    update!(p.learner, t, e, s)
end

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
