export StateOverriddenObs

using MacroTools: @forward

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
