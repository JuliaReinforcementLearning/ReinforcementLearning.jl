export StateOverriddenObs, BatchObs, RewardOverriddenObs

using MacroTools: @forward

#####
# StateOverriddenObs
#####

"""
    StateOverriddenObs(;obs, state)

Replace the internal state of `obs` with `state`.

# Example

```julia-repl
julia> old_obs = (reward=1.0, terminal=false, state=1)
(reward = 1.0, terminal = false, state = 1)

julia> new_obs = StateOverriddenObs(;obs=old_obs, state=nothing)
StateOverriddenObs{NamedTuple{(:reward, :terminal, :state),Tuple{Float64,Bool,Int64}},Nothing}((reward = 1.0, terminal = false, state = 1), nothing)

julia> get_state(new_obs) === nothing
true

julia> get_reward(new_obs) === get_reward(old_obs)
true
```
"""
Base.@kwdef struct StateOverriddenObs{O,S}
    obs::O
    state::S
end

@forward StateOverriddenObs.obs ActionStyle,
get_legal_actions,
get_legal_actions_mask,
get_terminal,
get_reward

get_state(obs::StateOverriddenObs) = obs.state

#####
# BatchObs
#####

"""
    BatchObs(obs::Vector)

Wrap several observations into a batch.
"""
struct BatchObs{O}
    obs::Vector{O}
end

@forward BatchObs.obs Base.getindex, Base.length, Base.setindex!, Base.iterate

get_terminal(obs::BatchObs) = [get_terminal(x) for x in obs]
get_reward(obs::BatchObs) = [get_reward(x) for x in obs]

get_legal_actions(obs::BatchObs) = [get_legal_actions(x) for x in obs]

function get_legal_actions_mask(obs::BatchObs)
    first_mask = get_legal_actions_mask(obs[1])
    mask = similar(first_mask, length(first_mask), length(obs))
    for i in 1:length(obs)
        obs[:, i] .= get_legal_actions_mask(obs[i])
    end
    mask
end

function get_state(obs::BatchObs)
    first_state = get_state(obs[1])
    m, n = length(first_state), length(obs)
    state = similar(first_state, size(first_state)..., n)
    for i in 1:n
        state[m*(i-1)+1:m*i] .= get_state(obs[i]) |> vec
    end
    state
end

struct RewardOverriddenObs{O, R}
    obs::O
    reward::R
end

@forward RewardOverriddenObs.obs ActionStyle,
get_legal_actions,
get_legal_actions_mask,
get_terminal,
get_state

get_reward(obs::RewardOverriddenObs) = obs.reward