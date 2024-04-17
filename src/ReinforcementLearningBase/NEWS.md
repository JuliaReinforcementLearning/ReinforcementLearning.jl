### ReinforcementLearningBase.jl Release Notes

#### v0.12.0

- Transition to `RLCore.forward`, `RLBase.act!`, `RLBase.plan!` and `Base.push!` syntax instead of functional objects for hooks, policies and environments

#### v0.9.7

- Update POMDPModelTools -> POMDPTools
- Add `next_player!` method to support `Sequential` `MultiAgent` environments

#### v0.9.6

- Implement `Base.:(==)` for `Space`. [#428](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/428)

#### v0.9.5

- Add default `Base.:(==)` and `Base.hash` method for `AbstractEnv`. [#348](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/348)