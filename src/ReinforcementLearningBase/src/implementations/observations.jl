export StateOverriddenObs

using MacroTools: @forward

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
