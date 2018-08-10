"""
    struct MDPLearner
        γ::Float64
        policy::Array{Int64, 1}
        values::Array{Float64, 1}
        mdp::MDP

Used to solve `mdp` with discount factor `γ`.
"""
@with_kw struct MDPLearner
    mdp::MDP = MDP()
    γ::Float64 = .9
    policy::Array{Int64, 1} = ones(Int64, mdp.ns)
    values::Array{Float64, 1} = zeros(mdp.ns)
end
export MDPLearner

function MDPLearner(mdp, γ::Float64)
    return MDPLearner(γ = γ, mdp = mdp)
end

@inline function selectaction(learner::MDPLearner, 
                              policy::AbstractEpsilonGreedyPolicy, state)
    if rand() < policy.ϵ
        rand(1:learner.mdp.na)
    else
        learner.policy[state]
    end
end

# solve MDP

function get_optimal_policy_given_values!(mdplearner::MDPLearner)
    for state in findall(x -> x == 0, mdplearner.mdp.isterminal)
        mdplearner.policy[state], vmax = argmaxvalue(mdplearner, state)
    end
    return mdplearner.policy
end

function argmaxvalue(mdplearner, state)
    amax = 0; vmax = -Inf64
    for a in 1:mdplearner.mdp.na
        v = mdplearner.mdp.reward[a, state] + mdplearner.γ *
                dot(mdplearner.mdp.trans_probs[a, state], mdplearner.values)
        if vmax < v 
            vmax = v
            amax = a
        end
    end
    amax, vmax
end
        
function geteffectivetandr(mdplearner)
    trans_probs = []
    reward = zeros(mdplearner.mdp.ns)
    for state = 1:mdplearner.mdp.ns
        if size(mdplearner.mdp.trans_probs, 2) < state ||
            mdplearner.mdp.isterminal[state] == 1
            push!(trans_probs, SparseVector(mdplearner.mdp.ns, Int64[], Float64[]))
        else
            push!(trans_probs, mdplearner.mdp.trans_probs[mdplearner.policy[state], state])
        end
        reward[state] = mdplearner.mdp.reward[mdplearner.policy[state], state]
    end
    hcat(trans_probs...), reward
end

function get_values_given_policy!(mdplearner::MDPLearner)
    trans_probs, reward = geteffectivetandr(mdplearner)
    mdplearner.values[:] = get_value(reward, trans_probs, mdplearner.γ)
end

"""
    policy_iteration!(mdplearner::MDPLearner)

Solve MDP with policy iteration using [`MDPLearner`](@ref).
"""
function policy_iteration!(mdplearner::MDPLearner)
    oldpolicy = zeros(mdplearner.mdp.ns)
    while sum(abs.(oldpolicy - mdplearner.policy)) > 0
        oldpolicy[:] = mdplearner.policy[:]
        get_values_given_policy!(mdplearner)
        get_optimal_policy_given_values!(mdplearner)
    end
end
export policy_iteration!

function value_iteration!(mdplearner::MDPLearner; eps = 1.e-8)
    diff = 1.
    values = zeros(mdplearner.mdp.ns)
    while diff > eps
        for state in findall(x -> x == 0, mdplearner.mdp.isterminal)
            amax, values[state] = argmaxvalue(mdplearner, state)
        end
        diff = norm(values - mdplearner.values)
        mdplearner.values[:] = values[:]
    end
end
export value_iteration!

# utilities

function get_Q_values(mdplearner::MDPLearner)
    [mdplearner.mdp.reward[action, state] + mdplearner.γ * (transpose(mdplearner.values) * mdplearner.mdp.trans_probs[:, action, state])[1,1] for action = 1:mdplearner.mdp.na, state = 1:mdplearner.mdp.ns]
end

function get_value(reward, trans_probs, γ)
    return (sparse(Matrix(1.0I, length(reward), length(reward))) - 
            γ * transpose(trans_probs)) \ reward
end

update!(::MDPLearner) = Nothing
