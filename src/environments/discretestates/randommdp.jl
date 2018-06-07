using ReinforcementLearning

function DetMDP(; ns = 10^4, na = 10)
    mdp = MDP(ns, na, init = "deterministic")
    mdp.reward = mdp.reward .* (mdp.reward .< -1.5)
    mdp
end

function DetTreeMDP(; na = 4, depth = 5)
    mdp = treeMDP(na, depth, init = "deterministic")
end

function DetTreeMDPwithinrew(; args...)
    mdp = DetTreeMDP(; args...)
    nonterminals = find(1 - mdp.isterminal)
    mdp.reward[:, nonterminals] = -rand(mdp.na, length(nonterminals))
    mdp
end

function StochTreeMDP(; na = 4, depth = 4, bf = 2)
    mdp = treeMDP(na, depth, init = "random", branchingfactor = bf)
    mdp
end

StochMDP(; na = 10, ns = 50) = MDP(ns, na)

function AbsorbingDetMDP(;ns = 10^3, na = 10)
    mdp = MDP(ns, na, init = "deterministic")
    mdp.reward .= mdp.reward .* (mdp.reward .< -.5)
    mdp.initialstates = 1:div(ns, 100)
    reset!(mdp)
    setterminalstates!(mdp, ns - div(ns, 100) + 1:ns)
    mdp
end

