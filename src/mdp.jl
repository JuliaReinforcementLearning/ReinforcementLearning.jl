"""
    mutable struct MDP 
        ns::Int64
        na::Int64
        state::Int64
        trans_probs::Array{AbstractArray, 2}
        reward::Array{Float64, 2}
        initialstates::Array{Int64, 1}
        isterminal::Array{Int64, 1}
    
A Markov Decision Process with `ns` states, `na` actions, current `state`,
`na`x`ns` - array of transition probabilites `trans_props` which consists for
every (action, state) pair of a (potentially sparse) array that sums to 1 (see
[`getprobvecrandom`](@ref), [`getprobvecuniform`](@ref),
[`getprobvecdeterministic`](@ref) for helpers to constract the transition
probabilities) `na`x`ns` - array of `reward`, array of initial states
`initialstates`, and `ns` - array of 0/1 indicating if a state is terminal.
"""
mutable struct MDP 
    ns::Int64
    na::Int64
    state::Int64
    trans_probs::Array{AbstractArray, 2}
    reward::Array{Float64, 2}
    initialstates::Array{Int64, 1}
    isterminal::Array{Int64, 1}
end
export MDP

function interact!(action, env::MDP)
    r = env.reward[action, env.state]
    run!(env, action)
    env.state, r, env.isterminal[env.state] == 1
end
function getstate(env::MDP)
    env.state, env.isterminal[env.state] == 1
end
function reset!(env::MDP)
    env.state = rand(env.initialstates)
end

"""
    getprobvecrandom(n) 

Returns an array of length `n` that sums to 1. More precisely, the array is a
sample of a [Dirichlet
distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution) with `n`
categories and ``\alpha_1 = \cdots =\alpha_n = 1``.
"""
getprobvecrandom(n) = normalize(rand(n), 1)
"""
    getprobvecrandom(n, min, max)

Returns an array of length `n` that sums to 1 where all elements outside of
`min`:`max` are zero.
"""
getprobvecrandom(n, min, max) = SparseVector(n, collect(min:max),
                                             getprobvecrandom(max - min + 1))
"""
    getprobvecuniform(n)  = fill(1/n, n)
"""
getprobvecuniform(n) = fill(1/n, n) 
"""
    getprobvecdeterministic(n, min = 1, max = n)

Returns a `SparseVector` of length `n` where one element in `min`:`max` has 
value 1.
"""
getprobvecdeterministic(n, min = 1, max = n) = SparseVector(n, [rand(min:max)], [1.])
# constructors
"""
    MDP(ns, na; init = "random")
    MDP(; ns = 10, na = 4, init = "random")

Return MDP with `init in ("random", "uniform", "deterministic")`, where the
keyword init determines how to construct the transition probabilites (see also 
[`getprobvecrandom`](@ref), [`getprobvecuniform`](@ref),
[`getprobvecdeterministic`](@ref)).
"""
function MDP(ns, na; init = "random")
    r = randn(na, ns)
    func = eval(parse("getprobvec" * init))
    T = [func(ns) for a in 1:na, s in 1:ns]
    MDP(ns, na, rand(1:ns), T, r,
        1:ns, zeros(ns))
end
function MDP(; ns = 10, na = 4, init = "random")
    MDP(ns, na, init = init)
end

"""
    treeMDP(na, depth; init = "random", branchingfactor = 3)

Returns a tree structured MDP with na actions and `depth` of the tree.
If `init` is random, the `branchingfactor` determines how many possible states a
(action, state) pair has. If `init = "deterministic"` the `branchingfactor =
na`.
"""
function treeMDP(na, depth; 
                 init = "random", 
                 branchingfactor = 3)
    isdet = (init == "deterministic")
    if isdet
        branchingfactor = na
        ns = na.^(0:depth - 1)
    else
        ns = branchingfactor.^(0:depth - 1)
    end
    cns = cumsum(ns)
    func = eval(parse("getprobvec" * init))
    T = Array{SparseVector, 2}(na, cns[end])
    for i in 1:depth - 1
        for s in 1:ns[i]
            for a in 1:na
                lb = cns[i] + (s - 1) * branchingfactor + (a - 1) * isdet + 1
                ub = isdet ? lb : lb + branchingfactor - 1
                T[a, (i == 1 ? 0 : cns[i-1]) + s] = func(cns[end] + 1, lb, ub)
            end
        end
    end
    r = zeros(na, cns[end] + 1)
    isterminal = [zeros(cns[end]); 1]
    for s in cns[end-1]+1:cns[end]
        for a in 1:na
            r[a, s] = -rand()
            T[a, s] = getprobvecdeterministic(cns[end] + 1, cns[end] + 1,
                                              cns[end] + 1)
        end
    end
    MDP(cns[end] + 1, na, 1, T, r, 1:1, isterminal)
end
export treeMDP

function emptytransprob!(v::SparseVector)
    empty!(v.nzind); empty!(v.nzval)
end
emptytransprob!(v::Array{Float64, 1}) = v[:] .*= 0.

"""
    setterminalstates!(mdp, range)

Sets `mdp.isterminal[range] .= 1`, empties the table of transition probabilities
for terminal states and sets the reward for all actions in the terminal state to
the same value.
"""
function setterminalstates!(mdp, range)
    mdp.isterminal[range] .= 1
    for s in find(mdp.isterminal)
        mdp.reward[:, s] .= mean(mdp.reward[:, s])
        for a in 1:mdp.na
            emptytransprob!(mdp.trans_probs[a, s])
        end
    end
end
export setterminalstates!

# run MDP

function sample(w::Array{Float64, 1})
    r = rand()
    c = w[1]
    n = length(w)
    @inbounds for i in 1:n
        if c > r
            return i
        end
        c += w[i]
    end
    return n
end
function sample(w::SparseVector)
    w.nzind[sample(w.nzval)]
end

"""
    run!(mdp::MDP, action::Int64)

Transition to a new state given `action`. Returns the new state.
"""
function run!(mdp::MDP, action::Int64)
    if mdp.isterminal[mdp.state] == 1
        reset!(mdp)
    else
        mdp.state = sample(mdp.trans_probs[action, mdp.state])
    end
end

"""
    run!(mdp::MDP, policy::Array{Int64, 1}) = run!(mdp, policy[mdp.state])

"""
run!(mdp::MDP, policy::Array{Int64, 1}) = run!(mdp, policy[mdp.state])


