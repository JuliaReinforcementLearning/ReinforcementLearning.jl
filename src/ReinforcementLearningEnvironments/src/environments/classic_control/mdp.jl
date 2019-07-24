using Random, POMDPs, POMDPModels, SparseArrays, LinearAlgebra, StatsBase

export MDPEnv, POMDPEnv, SimpleMDPEnv, absorbing_deterministic_tree_MDP, stochastic_MDP, stochastic_tree_MDP,
    deterministic_tree_MDP_with_rand_reward, deterministic_tree_MDP, deterministic_MDP,
    DeterministicStateActionReward, DeterministicNextStateReward,
    NormalStateActionReward, NormalNextStateReward

#####
##### POMDPEnv
#####

mutable struct POMDPEnv{T,Ts,Ta, R<:AbstractRNG}
    model::T
    state::Ts
    actions::Ta
    action_space::DiscreteSpace
    observation_space::DiscreteSpace
    rng::R
end

POMDPEnv(model; rng=Random.GLOBAL_RNG) = POMDPEnv(
    model,
    initialstate(model, rng),
    actions(model),
    DiscreteSpace(n_actions(model)),
    DiscreteSpace(n_states(model)),
    rng)

function interact!(env::POMDPEnv, action)
    s, o, r = generate_sor(env.model, env.state, env.actions[action], env.rng)
    env.state = s
    (observation = observationindex(env.model, o),
     reward = r,
     isdone = isterminal(env.model, s))
end

function observe(env::POMDPEnv)
    (observation = observationindex(env.model, generate_o(env.model, env.state, env.rng)),
     isdone = isterminal(env.model, env.state))
end

#####
##### MDPEnv
#####

mutable struct MDPEnv{T, Ts, Ta, R<:AbstractRNG}
    model::T
    state::Ts
    actions::Ta
    action_space::DiscreteSpace
    observation_space::DiscreteSpace
    rng::R
end

MDPEnv(model; rng=Random.GLOBAL_RNG) = MDPEnv(
    model,
    initialstate(model, rng),
    actions(model),
    DiscreteSpace(n_actions(model)),
    DiscreteSpace(n_states(model)),
    rng)

action_space(env::Union{MDPEnv, POMDPEnv}) = env.action_space
observation_space(env::Union{MDPEnv, POMDPEnv}) = env.observation_space

observationindex(env, o) = Int64(o) + 1

function reset!(env::Union{POMDPEnv, MDPEnv})
    initialstate(env.model, env.rng)
    nothing
end

function interact!(env::MDPEnv, action)
    s = rand(env.rng, transition(env.model, env.state, env.actions[action]))
    r = reward(env.model, env.state, env.actions[action])
    env.state = s
    (observation = stateindex(env.model, s),
     reward = r,
     isdone = isterminal(env.model, s))
end

function observe(env::MDPEnv)
    (observation = stateindex(env.model, env.state),
     isdone = isterminal(env.model, env.state))
end

#####
##### SimpleMDPEnv
#####
"""
    mutable struct SimpleMDPEnv
        ns::Int64
        na::Int64
        state::Int64
        trans_probs::Array{AbstractArray, 2}
        reward::R
        initialstates::Array{Int64, 1}
        isterminal::Array{Int64, 1}
        rng::S
A Markov Decision Process with `ns` states, `na` actions, current `state`,
`na`x`ns` - array of transition probabilites `trans_props` which consists for
every (action, state) pair of a (potentially sparse) array that sums to 1 (see
[`get_prob_vec_random`](@ref), [`get_prob_vec_uniform`](@ref),
[`get_prob_vec_deterministic`](@ref) for helpers to constract the transition
probabilities) `reward` of type `R` (see [`DeterministicStateActionReward`](@ref),
[`DeterministicNextStateReward`](@ref), [`NormalNextStateReward`](@ref),
[`NormalStateActionReward`](@ref)), array of initial states
`initialstates`, and `ns` - array of 0/1 indicating if a state is terminal.
"""
mutable struct SimpleMDPEnv{T,R,S<:AbstractRNG}
    observation_space::DiscreteSpace
    action_space::DiscreteSpace
    state::Int64
    trans_probs::Array{T, 2}
    reward::R
    initialstates::Array{Int64, 1}
    isterminal::Array{Int64, 1}
    rng::S
end

function SimpleMDPEnv(ospace, aspace, state, trans_probs::Array{T, 2},
                      reward::R, initialstates, isterminal,
                      rng::S = Random.GLOBAL_RNG) where {T,R,S}
    if R <: AbstractMatrix # to ensure compatibility with previous versions
        reward = DeterministicStateActionReward(reward)
    end
    SimpleMDPEnv{T,typeof(reward),S}(ospace, aspace, state, trans_probs,
                                     reward, initialstates, isterminal, rng)
end

observation_space(env::SimpleMDPEnv) = env.observation_space
action_space(env::SimpleMDPEnv) = env.action_space

# reward types
"""
    struct DeterministicNextStateReward
        value::Vector{Float64}
"""
struct DeterministicNextStateReward
    value::Vector{Float64}
end
reward(::AbstractRNG, r::DeterministicNextStateReward, s, a, s′) = r.value[s′]
expected_rewards(r::DeterministicNextStateReward, trans_probs) = expected_rewards(r.value, trans_probs)
function expected_rewards(r::AbstractVector, trans_probs)
    result = zeros(size(trans_probs))
    for i in eachindex(trans_probs)
        result[i] = dot(trans_probs[i], r)
    end
    result
end
"""
    struct DeterministicStateActionReward
        value::Array{Float64, 2}

`value` should be a `na × ns`-matrix.
"""
struct DeterministicStateActionReward
    value::Array{Float64, 2}
end
reward(::AbstractRNG, r::DeterministicStateActionReward, s, a, s′) = r.value[a, s]
expected_rewards(r::DeterministicStateActionReward, ::Any) = r.value
"""
    struct NormalNextStateReward
        mean::Vector{Float64}
        std::Vector{Float64}
"""
struct NormalNextStateReward
    mean::Vector{Float64}
    std::Vector{Float64}
end
reward(rng, r::NormalNextStateReward, s, a, s′) = r.mean[s′] + randn(rng) * r.std[s′]
expected_rewards(r::NormalNextStateReward, trans_probs) = expected_rewards(r.mean, trans_probs)
"""
    struct NormalStateActionReward
        mean::Array{Float64, 2}
        std::Array{Float64, 2}

`mean` and `std` should be `na × ns`-matrices.
"""
struct NormalStateActionReward
    mean::Array{Float64, 2}
    std::Array{Float64, 2}
end
reward(rng, r::NormalStateActionReward, s, a, s′) = r.mean[a, s] + randn(rng) * r.std[a, s]
expected_rewards(r::NormalStateActionReward, ::Any) = r.mean

# run SimpleMDPEnv
"""
    run!(mdp::SimpleMDPEnv, action::Int64)
Transition to a new state given `action`. Returns the new state.
"""
function run!(mdp::SimpleMDPEnv, action::Int64)
    if mdp.isterminal[mdp.state] == 1
        reset!(mdp)
    else
        mdp.state = wsample(mdp.rng, mdp.trans_probs[action, mdp.state])
        (observation = mdp.state,)
    end
end

"""
    run!(mdp::SimpleMDPEnv, policy::Array{Int64, 1}) = run!(mdp, policy[mdp.state])
"""
run!(mdp::SimpleMDPEnv, policy::Array{Int64, 1}) = run!(mdp, policy[mdp.state])


function interact!(env::SimpleMDPEnv, action)
    oldstate = env.state
    run!(env, action)
    r = reward(env.rng, env.reward, oldstate, action, env.state)
    (observation = env.state, reward = r, isdone = env.isterminal[env.state] == 1)
end

function observe(env::SimpleMDPEnv)
    (observation = env.state, isdone = env.isterminal[env.state] == 1)
end

function reset!(env::SimpleMDPEnv)
    env.state = rand(env.rng, env.initialstates)
    nothing
end

"""
    get_prob_vec_random(n)
Returns an array of length `n` that sums to 1. More precisely, the array is a
sample of a [Dirichlet
distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution) with `n`
categories and ``α_1 = ⋯  = α_n = 1``.
"""
get_prob_vec_random(n) = normalize(rand(n), 1)
"""
    get_prob_vec_random(n, min, max)
Returns an array of length `n` that sums to 1 where all elements outside of
`min`:`max` are zero.
"""
get_prob_vec_random(n, min, max) = sparsevec(collect(min:max),
                                          get_prob_vec_random(max - min + 1), n)
"""
    get_prob_vec_uniform(n)  = fill(1/n, n)
"""
get_prob_vec_uniform(n) = fill(1/n, n)
"""
    get_prob_vec_deterministic(n, min = 1, max = n)
Returns a `SparseVector` of length `n` where one element in `min`:`max` has
value 1.
"""
get_prob_vec_deterministic(n, min = 1, max = n) = sparsevec([rand(min:max)], [1.], n)
# constructors
"""
    SimpleMDPEnv(ns, na; init = "random")
    SimpleMDPEnv(; ns = 10, na = 4, init = "random")
Return SimpleMDPEnv with `init in ("random", "uniform", "deterministic")`, where the
keyword init determines how to construct the transition probabilites (see also
[`get_prob_vec_random`](@ref), [`get_prob_vec_uniform`](@ref),
[`get_prob_vec_deterministic`](@ref)).
"""
function SimpleMDPEnv(ns, na; init = "random")
    r = randn(na, ns)
    func = eval(Symbol("get_prob_vec_" * init))
    T = [func(ns) for a in 1:na, s in 1:ns]
    SimpleMDPEnv(DiscreteSpace(ns), DiscreteSpace(na), rand(1:ns), T, r,
        1:ns, zeros(ns))
end

SimpleMDPEnv(; ns = 10, na = 4, init = "random") = SimpleMDPEnv(ns, na, init = init)


"""
    tree_MDP(na, depth; init = "random", branchingfactor = 3)
Returns a tree structured SimpleMDPEnv with na actions and `depth` of the tree.
If `init` is random, the `branchingfactor` determines how many possible states a
(action, state) pair has. If `init = "deterministic"` the `branchingfactor =
na`.
"""
function tree_MDP(na, depth;
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
    func = eval(Symbol("get_prob_vec_" * init))
    T = Array{SparseVector, 2}(undef, na, cns[end])
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
            T[a, s] = get_prob_vec_deterministic(cns[end] + 1, cns[end] + 1,
                                              cns[end] + 1)
        end
    end
    SimpleMDPEnv(DiscreteSpace(cns[end] + 1), DiscreteSpace(na), 1, T, r, 1:1, isterminal)
end

function empty_trans_prob!(v::SparseVector)
    empty!(v.nzind)
    empty!(v.nzval)
end
empty_trans_prob!(v::Array{Float64, 1}) = v[:] .*= 0.

"""
    set_terminal_states!(mdp, range)
Sets `mdp.isterminal[range] .= 1`, empties the table of transition probabilities
for terminal states and sets the reward for all actions in the terminal state to
the same value.
"""
function set_terminal_states!(mdp, range)
    mdp.isterminal[range] .= 1
    for s in findall(x -> x == 1, mdp.isterminal)
        mdp.reward[:, s] .= mean(mdp.reward[:, s])
        for a in 1:length(mdp.action_space)
            empty_trans_prob!(mdp.trans_probs[a, s])
        end
    end
end

"""
    deterministic_MDP(; ns = 10^4, na = 10)
Returns a random deterministic SimpleMDPEnv.
"""
function deterministic_MDP(; ns = 10^4, na = 10)
    mdp = SimpleMDPEnv(ns, na, init = "deterministic")
    mdp.reward = mdp.reward .* (mdp.reward .< -1.5)
    mdp
end

"""
    deterministic_tree_MDP(; na = 4, depth = 5)
Returns a tree_MDP with random rewards at the leaf nodes.
"""
function deterministic_tree_MDP(; na = 4, depth = 5)
    mdp = tree_MDP(na, depth, init = "deterministic")
end

"""
    deterministic_tree_MDP_with_rand_reward(; na = 4, depth = 5)
Returns a tree_MDP with random rewards.
"""
function deterministic_tree_MDP_with_rand_reward(; args...)
    mdp = deterministic_tree_MDP(; args...)
    nonterminals = findall(x -> x == 0, mdp.isterminal)
    mdp.reward[:, nonterminals] = -rand(length(mdp.action_space), length(nonterminals))
    mdp
end

"""
    stochastic_tree_MDP(; na = 4, depth = 4, bf = 2)
Returns a random stochastic tree_MDP with branching factor `bf`.
"""
function stochastic_tree_MDP(; na = 4, depth = 4, bf = 2)
    mdp = tree_MDP(na, depth, init = "random", branchingfactor = bf)
    mdp
end

"""
    stochastic_MDP(; na = 10, ns = 50) = SimpleMDPEnv(ns, na)
"""
stochastic_MDP(; na = 10, ns = 50) = SimpleMDPEnv(ns, na)

"""
    absorbing_deterministic_tree_MDP(;ns = 10^3, na = 10)
Returns a random deterministic absorbing SimpleMDPEnv
"""
function absorbing_deterministic_tree_MDP(;ns = 10^3, na = 10)
    mdp = SimpleMDPEnv(ns, na, init = "deterministic")
    mdp.reward .= mdp.reward .* (mdp.reward .< -.5)
    mdp.initialstates = 1:div(ns, 100)
    reset!(mdp)
    set_terminal_states!(mdp, ns - div(ns, 100) + 1:ns)
    mdp
end
