"""
    DetMDP(; ns = 10^4, na = 10)

Returns a random deterministic MDP.
"""
function DetMDP(; ns = 10^4, na = 10)
    mdp = MDP(ns, na, init = "deterministic")
    mdp.reward = mdp.reward .* (mdp.reward .< -1.5)
    mdp
end

"""
    DetTreeMDP(; na = 4, depth = 5) 

Returns a treeMDP with random rewards at the leaf nodes.
"""
function DetTreeMDP(; na = 4, depth = 5)
    mdp = treeMDP(na, depth, init = "deterministic")
end

"""
    DetTreeMDPwithinrew(; na = 4, depth = 5)

Returns a treeMDP with random rewards.
"""
function DetTreeMDPwithinrew(; args...)
    mdp = DetTreeMDP(; args...)
    nonterminals = findall(x -> x == 0, mdp.isterminal)
    mdp.reward[:, nonterminals] = -rand(mdp.na, length(nonterminals))
    mdp
end

"""
    StochTreeMDP(; na = 4, depth = 4, bf = 2)

Returns a random stochastic treeMDP with branching factor `bf`.
"""
function StochTreeMDP(; na = 4, depth = 4, bf = 2)
    mdp = treeMDP(na, depth, init = "random", branchingfactor = bf)
    mdp
end

"""
    StochMDP(; na = 10, ns = 50) = MDP(ns, na)
"""
StochMDP(; na = 10, ns = 50) = MDP(ns, na)

"""
    AbsorbingDetMDP(;ns = 10^3, na = 10)

Returns a random deterministic absorbing MDP
"""
function AbsorbingDetMDP(;ns = 10^3, na = 10)
    mdp = MDP(ns, na, init = "deterministic")
    mdp.reward .= mdp.reward .* (mdp.reward .< -.5)
    mdp.initialstates = 1:div(ns, 100)
    reset!(mdp)
    setterminalstates!(mdp, ns - div(ns, 100) + 1:ns)
    mdp
end

export DetMDP, DetTreeMDP, DetTreeMDPwithinrew, StochMDP, StochTreeMDP, AbsorbingDetMDP
