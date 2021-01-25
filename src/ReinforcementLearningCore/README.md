# ReinforcementLearningCore

[![CI](https://github.com/JuliaReinforcementLearning/ReinforcementLearningCore.jl/workflows/CI/badge.svg)](https://github.com/JuliaReinforcementLearning/ReinforcementLearningCore.jl/actions?query=workflow%3ACI)
[![CodeCoverage](https://codecov.io/gh/JuliaReinforcementLearning/ReinforcementLearningCore.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaReinforcementLearning/ReinforcementLearningCore.jl)

This package is the core component of [ReinforcementLearning.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl). It provides some typical implementations of the interfaces defined in [ReinforcementLearningBase.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl).


## Code Structure

```
./src
├── core (define how policies interact with environments)
├── extensions (patch code fore upstream packages are stored here)
├── policies (all policies are put here)
│   ├── agents (= policy + trajectory)
│   ├── q_based_policies
│   │   ├── explorers (select action from action-values)
│   │   └── learners (learn state-value, state-action-value, distributional value...)
│   │       └── approximators (= NN + Optimiser)
│   └── (some other common policies).jl
└── utils (Reusable functions/structures)
```

For developers who are interested in contributing, I suggest you read the source code in the following top-down order:

- `core/run.jl`
- `policies/base.jl`
- `policies/agents/agent.jl`
- `policies/agents/trajectories/trajectory.jl`
- `policies/q_based_policies/q_based_policy.jl`
- `policies/q_based_policies/learners/approximators/neural_network_approximator.jl`
- `policies/q_based_policies/explorers/weighted_explorer.jl`

Then play with the ``E`JuliaRL_BasicDQN_CartPole` `` experiment in [ReinforcementLearningZoo.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl) and try to understand the runtime logic step by step. After that, you can explore other components on demand.
