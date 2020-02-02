export StateOverriddenObs, ObsAndAction

using MacroTools: @forward

struct StateOverriddenObs{O,S}
    obs::O
    state::S
end

@forward StateOverriddenObs.obs RLBase.ActionStyle,
RLBase.legal_actions,
RLBase.legal_actions_mask,
RLBase.get_terminal,
RLBase.get_reward

RLBase.get_state(obs::StateOverriddenObs) = obs.state

struct ObsAndAction{O,A}
    obs::O
    action::A
end

@forward ObsAndAction.obs RLBase.ActionStyle,
RLBase.legal_actions,
RLBase.legal_actions_mask,
RLBase.get_terminal,
RLBase.get_reward,
RLBase.get_state

get_action(obs::ObsAndAction) = obs.action
