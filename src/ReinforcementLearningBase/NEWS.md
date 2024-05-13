### ReinforcementLearningBase.jl Release Notes

#### v0.13.1

- Don't call `legal_action_space_mask` methods when `ActionStyle` is `MinimalActionSet`

#### v0.13.0

- Breaking release compatible with RL.jl v0.11

#### v0.12.0

- Transition to `RLCore.forward`, `RLBase.act!`, `RLBase.plan!` and `Base.push!` syntax instead of functional objects for hooks, policies and environments

#### v0.9.7

- Update POMDPModelTools -> POMDPTools
- Add `next_player!` method to support `Sequential` `MultiAgent` environments

#### v0.9.6

- Implement `Base.:(==)` for `Space`. [#428](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/428)

#### v0.9.5

- Add default `Base.:(==)` and `Base.hash` method for `AbstractEnv`. [#348](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/348)